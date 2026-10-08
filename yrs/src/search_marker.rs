//! Search markers, ported from Yjs `AbstractType`.
//!
//! A marker remembers a block and the visible index where that block starts. The next
//! index lookup starts at the nearest marker and walks only the gap, instead of walking
//! from the first block.
//!
//! Markers are a hint that must stay exact. A stale index sends the following insert to
//! the wrong block. Local inserts and deletes adjust every marker's index. Squashing a
//! block into its left neighbor retargets markers that pointed at the discarded block.
//! A remote transaction drops the list: remote items are integrated by id, not by index,
//! so there is no delta to apply until the next local lookup rebuilds the list.
//!
//! Lookups that need the open formatting attributes ([`crate::Text::insert_with_attributes`],
//! [`crate::Text::format`]) still walk from the start. Yjs does the same (`findPosition`
//! with `useSearchMarker = !attributes`), because the attribute map is the set of format
//! items to the left of the index. Those calls still update marker indexes, so a later
//! plain insert can use the cache.

use crate::block::ItemPtr;
use crate::branch::BranchPtr;
use crate::doc::OffsetKind;
use crate::transaction::ReadTxn;
use std::sync::atomic::{AtomicU32, Ordering};

const MAX_SEARCH_MARKERS: u32 = 80;

static MARKER_CLOCK: AtomicU32 = AtomicU32::new(1);

#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchMarker {
    ptr: ItemPtr,
    /// Visible index of the first countable character of [`Self::ptr`].
    index: u32,
    timestamp: u32,
}

fn fresh_timestamp() -> u32 {
    MARKER_CLOCK.fetch_add(1, Ordering::Relaxed)
}

/// Drop every marker.
///
/// Yjs does this from `_callObserver` when `!transaction.local`. Remote items
/// are integrated by id, so a visible index would be a guess. Local edits,
/// including [crate::Text::apply_delta], update indexes instead.
pub(crate) fn clear(mut branch: BranchPtr) {
    branch.search_markers.clear();
}

/// Read-only: the rightmost marker whose index is at or before `index`.
/// Does not create or move markers, so it is safe under a read transaction.
pub(crate) fn hint(branch: BranchPtr, index: u32) -> Option<(ItemPtr, u32)> {
    let mut best: Option<&SearchMarker> = None;
    for marker in branch.search_markers.iter() {
        if marker.index <= index {
            let closer = best.map(|b| marker.index >= b.index).unwrap_or(true);
            if closer {
                best = Some(marker);
            }
        }
    }
    best.map(|marker| (marker.ptr, marker.index))
}

/// Walk to the block that contains `index`, remembering that spot for the next lookup.
///
/// Returns `(block, visible index of that block's start)`. The remaining
/// `index - returned_index` characters sit inside the block or on the format
/// items directly in front of the next countable one.
pub(crate) fn track<T: ReadTxn>(
    mut branch: BranchPtr,
    txn: &T,
    index: u32,
) -> Option<(ItemPtr, u32)> {
    if branch.start.is_none() || index == 0 {
        return None;
    }
    let kind = txn.store().offset_kind;
    remember_kind(branch, kind);

    let mut p = branch.start.unwrap();
    let mut pindex = 0u32;
    let mut reused: Option<usize> = None;

    if !branch.search_markers.is_empty() {
        let mut best_i = 0usize;
        let mut best_diff = u32::MAX;
        for (i, marker) in branch.search_markers.iter().enumerate() {
            let diff = index.abs_diff(marker.index);
            if diff < best_diff {
                best_diff = diff;
                best_i = i;
            }
        }
        p = branch.search_markers[best_i].ptr;
        pindex = branch.search_markers[best_i].index;
        branch.search_markers[best_i].timestamp = fresh_timestamp();
        reused = Some(best_i);
    }

    while p.right.is_some() && pindex < index {
        if !p.is_deleted() && p.is_countable() {
            let len = p.content_len(kind);
            if index < pindex.saturating_add(len) {
                break;
            }
            pindex = pindex.saturating_add(len);
        }
        p = p.right.unwrap();
    }

    while pindex > index {
        let Some(left) = p.left else {
            break;
        };
        p = left;
        if !p.is_deleted() && p.is_countable() {
            pindex = pindex.saturating_sub(p.content_len(kind));
        }
    }

    // Stay on the block that contains `index`. Yjs `findMarker` walks left across
    // a same-client, consecutive-clock run so the marker is not left on a block
    // that commit will free. `on_squash` retargets the marker when that block is
    // dropped. Walking left while the blocks are still alive sends the next
    // lookup back to the start of the run.

    if pindex > index || p.is_deleted() || !p.is_countable() {
        return None;
    }

    if let Some(slot) = reused {
        let marker_index = branch.search_markers[slot].index;
        let moved = marker_index.abs_diff(pindex);
        let limit = (branch.content_len / MAX_SEARCH_MARKERS).max(1);
        if moved < limit {
            overwrite(branch, slot, p, pindex);
            return Some((p, pindex));
        }
    }
    remember(branch, p, pindex);
    Some((p, pindex))
}

/// Shift marker indexes around a local insert (`len > 0`) or delete (`len < 0`).
///
/// `index` is the visible index where the change starts, counted before the
/// change. Yjs `updateMarkerChanges` takes the same pair. A delete in Yjs
/// records it after walking the removed items, but that walk does not advance
/// the index, so the value is still the start of the deletion.
pub(crate) fn note_visible_change(mut branch: BranchPtr, kind: OffsetKind, index: u32, len: i32) {
    if branch.search_markers.is_empty() || len == 0 {
        return;
    }
    remember_kind(branch, kind);

    let mut i = branch.search_markers.len();
    'markers: while i > 0 {
        i -= 1;
        if len > 0 {
            let mut ptr = branch.search_markers[i].ptr;
            let mut adjusted = branch.search_markers[i].index;
            while ptr.is_deleted() || !ptr.is_countable() {
                match ptr.left {
                    Some(left) => {
                        ptr = left;
                        if !ptr.is_deleted() && ptr.is_countable() {
                            adjusted = adjusted.saturating_sub(ptr.content_len(kind));
                            break;
                        }
                    }
                    None => {
                        branch.search_markers.remove(i);
                        continue 'markers;
                    }
                }
            }
            let duplicated = branch
                .search_markers
                .iter()
                .enumerate()
                .any(|(j, marker)| j != i && marker.ptr == ptr);
            if duplicated {
                branch.search_markers.remove(i);
                continue 'markers;
            }
            branch.search_markers[i].ptr = ptr;
            branch.search_markers[i].index = adjusted;
        }

        let marker_index = branch.search_markers[i].index;
        if index < marker_index || (len > 0 && index == marker_index) {
            let updated = marker_index as i64 + len as i64;
            branch.search_markers[i].index = (index as i64).max(updated).max(0) as u32;
        }
    }
}

/// `right` was merged into `left` and is about to be dropped. `delta` is the visible
/// length of `left` *before* the merge.
pub(crate) fn on_squash(mut branch: BranchPtr, left: ItemPtr, right: ItemPtr, delta: u32) {
    for marker in branch.search_markers.iter_mut() {
        if marker.ptr == right {
            marker.ptr = left;
            marker.index = marker.index.saturating_sub(delta);
        }
    }
}

pub(crate) fn forget_item(mut branch: BranchPtr, item: ItemPtr) {
    branch.search_markers.retain(|marker| marker.ptr != item);
}

fn remember_kind(mut branch: BranchPtr, kind: OffsetKind) {
    match branch.marker_offset_kind {
        Some(existing) if existing != kind => {
            branch.search_markers.clear();
            branch.marker_offset_kind = Some(kind);
        }
        None => branch.marker_offset_kind = Some(kind),
        Some(_) => {}
    }
}

fn overwrite(mut branch: BranchPtr, slot: usize, ptr: ItemPtr, index: u32) {
    branch.search_markers[slot].ptr = ptr;
    branch.search_markers[slot].index = index;
    branch.search_markers[slot].timestamp = fresh_timestamp();
}

fn remember(mut branch: BranchPtr, ptr: ItemPtr, index: u32) {
    if branch.search_markers.len() as u32 >= MAX_SEARCH_MARKERS {
        let slot = branch
            .search_markers
            .iter()
            .enumerate()
            .min_by_key(|(_, marker)| marker.timestamp)
            .map(|(i, _)| i)
            .unwrap();
        overwrite(branch, slot, ptr, index);
        return;
    }
    branch.search_markers.push(SearchMarker {
        ptr,
        index,
        timestamp: fresh_timestamp(),
    });
}

/// Every marker's stored index matches a walk from the start of the branch.
#[cfg(test)]
pub(crate) fn assert_consistent(branch: BranchPtr, kind: OffsetKind) {
    for marker in branch.search_markers.iter() {
        let mut current = branch.start;
        let mut index = 0u32;
        let mut found = false;
        while let Some(item) = current {
            if item == marker.ptr {
                assert_eq!(
                    index, marker.index,
                    "search marker drifted from the visible index"
                );
                found = true;
                break;
            }
            if !item.is_deleted() && item.is_countable() {
                index += item.content_len(kind);
            }
            current = item.right;
        }
        assert!(found, "search marker points outside the branch");
    }
}

#[cfg(test)]
mod test {
    use super::assert_consistent;
    use crate::branch::{Branch, BranchPtr};
    use crate::doc::{OffsetKind, Options};
    use crate::types::{Attrs, Delta};
    use crate::updates::decoder::Decode;
    use crate::{Any, Array, Doc, GetString, ReadTxn, Text, TextRef, Transact, Update};

    fn check(text: &TextRef, txn: &impl ReadTxn) {
        let kind = txn.store().offset_kind;
        let branch: &Branch = text.as_ref();
        assert_consistent(crate::branch::BranchPtr::from(branch), kind);
    }

    #[test]
    fn markers_survive_append_insert_and_delete() {
        let doc = Doc::new();
        let text = doc.get_or_insert_text("text");
        let mut txn = doc.transact_mut();
        for _ in 0..40 {
            let end = text.len(&txn);
            text.insert(&mut txn, end, "abcd");
        }
        assert_eq!(text.len(&txn), 160);
        check(&text, &txn);

        text.insert(&mut txn, 50, "XYZ");
        assert_eq!(text.get_string(&txn)[48..56], "abXYZcda".to_string());
        check(&text, &txn);

        text.remove_range(&mut txn, 20, 10);
        assert_eq!(text.len(&txn), 153);
        check(&text, &txn);

        let end = text.len(&txn);
        text.insert(&mut txn, end, "tail");
        assert!(text.get_string(&txn).ends_with("tail"));
        check(&text, &txn);
    }

    #[test]
    fn markers_follow_utf8_offsets() {
        let doc = Doc::new();
        let text = doc.get_or_insert_text("text");
        let mut txn = doc.transact_mut();
        text.insert(&mut txn, 0, "aaaa");
        text.insert(&mut txn, 4, "★★");
        text.insert(&mut txn, 0, "zz");
        // '★' is 3 bytes. "zz" + "aaaa" + "★★" => indexes 0,2,6.
        assert_eq!(text.get_string(&txn), "zzaaaa★★");
        assert_eq!(text.len(&txn), 6 + "★★".len() as u32);
        check(&text, &txn);
        text.insert(&mut txn, 6, "Q");
        assert_eq!(text.get_string(&txn), "zzaaaaQ★★");
        check(&text, &txn);
    }

    #[test]
    fn remote_update_drops_markers_and_local_insert_stays_correct() {
        let doc = Doc::new();
        let text = doc.get_or_insert_text("text");
        {
            let mut txn = doc.transact_mut();
            text.insert(&mut txn, 0, "hello");
            text.insert(&mut txn, 5, " world");
            check(&text, &txn);
        }
        let update = doc
            .transact()
            .encode_diff_v1(&crate::StateVector::default());

        let remote = Doc::new();
        let remote_text = remote.get_or_insert_text("text");
        remote
            .transact_mut()
            .apply_update(Update::decode_v1(&update).unwrap())
            .unwrap();
        {
            let mut txn = remote.transact_mut();
            remote_text.insert(&mut txn, 5, ",");
            assert_eq!(remote_text.get_string(&txn), "hello, world");
            check(&remote_text, &txn);
        }
    }

    #[test]
    fn attributed_insert_keeps_markers_consistent() {
        let doc = Doc::new();
        let text = doc.get_or_insert_text("text");
        let mut txn = doc.transact_mut();
        text.insert(&mut txn, 0, "hello");
        let bold = Attrs::from([("bold".into(), Any::Bool(true))]);
        text.insert_with_attributes(&mut txn, 5, "!", bold);
        assert_eq!(text.get_string(&txn), "hello!");
        check(&text, &txn);
        text.insert(&mut txn, 6, "x");
        assert_eq!(text.get_string(&txn), "hello!x");
        check(&text, &txn);
    }

    #[test]
    fn array_lookup_after_many_inserts() {
        let doc = Doc::new();
        let array = doc.get_or_insert_array("array");
        let mut txn = doc.transact_mut();
        for i in 0..50 {
            array.insert(&mut txn, i, i);
        }
        array.insert(&mut txn, 10, 1000);
        array.remove_range(&mut txn, 0, 5);
        assert_eq!(array.get(&txn, 0).unwrap().cast::<i64>().unwrap(), 5);
        assert_eq!(array.get(&txn, 5).unwrap().cast::<i64>().unwrap(), 1000);
        assert_eq!(array.len(&txn), 46);
        let branch: &Branch = array.as_ref();
        assert_consistent(BranchPtr::from(branch), txn.store().offset_kind);
    }

    #[test]
    fn markers_survive_commit_squash() {
        let doc = Doc::new();
        let text = doc.get_or_insert_text("text");
        {
            let mut txn = doc.transact_mut();
            for _ in 0..20 {
                let end = text.len(&txn);
                text.insert(&mut txn, end, "ab");
            }
            check(&text, &txn);
        }
        {
            let mut txn = doc.transact_mut();
            check(&text, &txn);
            text.insert(&mut txn, 10, "Q");
            assert_eq!(&text.get_string(&txn)[8..14], "abQaba");
            check(&text, &txn);
        }
    }

    #[test]
    fn format_leaves_marker_indexes_in_place() {
        let doc = Doc::new();
        let text = doc.get_or_insert_text("text");
        let mut txn = doc.transact_mut();
        text.insert(&mut txn, 0, "hello world");
        text.insert(&mut txn, 5, "X");
        let bold = Attrs::from([("bold".into(), Any::Bool(true))]);
        text.format(&mut txn, 0, 5, bold);
        assert_eq!(text.get_string(&txn), "helloX world");
        check(&text, &txn);
        let end = text.len(&txn);
        text.insert(&mut txn, end, "!");
        assert_eq!(text.get_string(&txn), "helloX world!");
        check(&text, &txn);
    }

    #[test]
    fn markers_follow_utf16_offsets() {
        let doc = Doc::with_options(Options {
            offset_kind: OffsetKind::Utf16,
            ..Default::default()
        });
        let text = doc.get_or_insert_text("text");
        let mut txn = doc.transact_mut();
        text.insert(&mut txn, 0, "aa★aa");
        text.insert(&mut txn, 4, "Z");
        check(&text, &txn);
        text.insert(&mut txn, 2, "Q");
        assert_eq!(text.get_string(&txn), "aaQ★aZa");
        assert_eq!(text.len(&txn), 7);
        check(&text, &txn);
    }

    #[test]
    fn apply_delta_updates_markers() {
        let doc = Doc::new();
        let text = doc.get_or_insert_text("text");
        let mut txn = doc.transact_mut();
        text.insert(&mut txn, 0, "hello");
        text.insert(&mut txn, 5, " world");
        check(&text, &txn);
        text.apply_delta(
            &mut txn,
            [Delta::retain(5), Delta::insert(","), Delta::retain(6)],
        );
        assert_eq!(text.get_string(&txn), "hello, world");
        check(&text, &txn);
        text.insert(&mut txn, 6, "X");
        assert_eq!(text.get_string(&txn), "hello,X world");
        check(&text, &txn);
    }
}

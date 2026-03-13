from rag_sgu.mlops import compute_dataset_fingerprint


def test_dataset_fingerprint_is_order_independent(tmp_path):
    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"
    file_a.write_bytes(b"alpha")
    file_b.write_bytes(b"beta")

    fp_1 = compute_dataset_fingerprint([file_a, file_b])
    fp_2 = compute_dataset_fingerprint([file_b, file_a])

    assert fp_1 == fp_2


def test_dataset_fingerprint_changes_on_file_update(tmp_path):
    file_a = tmp_path / "a.pdf"
    file_a.write_bytes(b"alpha")

    fp_1 = compute_dataset_fingerprint([file_a])
    file_a.write_bytes(b"alpha-updated")
    fp_2 = compute_dataset_fingerprint([file_a])

    assert fp_1 != fp_2

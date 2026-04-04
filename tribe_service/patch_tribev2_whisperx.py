from __future__ import annotations

from pathlib import Path


def main() -> None:
    target = Path("/usr/local/lib/python3.11/site-packages/tribev2/eventstransforms.py")

    source = target.read_text()

    if "# patched-by-pitchscore" in source:
        print("tribev2 whisperx patch already applied")
        return

    updated = source.replace(
        '        compute_type = "float16"\n',
        (
            '        compute_type = os.getenv("WHISPERX_CUDA_COMPUTE_TYPE", "float16") if device == "cuda" '
            'else os.getenv("WHISPERX_CPU_COMPUTE_TYPE", "float32")  # patched-by-pitchscore\n'
        ),
        1,
    )
    if updated == source:
        raise RuntimeError("Unable to patch tribev2 compute_type assignment")

    source = updated
    updated = source.replace(
        '                "16",\n',
        (
            '                str(int(os.getenv("WHISPERX_CUDA_BATCH_SIZE", "16")) if device == "cuda" '
            'else int(os.getenv("WHISPERX_CPU_BATCH_SIZE", "4"))),\n'
        ),
        1,
    )
    if updated == source:
        raise RuntimeError("Unable to patch tribev2 batch size assignment")

    source = updated

    target.write_text(source)
    print("patched", target)


if __name__ == "__main__":
    main()

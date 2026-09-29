"""Extract Video-MME's long videos from the downloaded zip chunks.

    python eval/videomme/extract_long_videos.py /dev/shm/videomme

eval/setup.sh videomme runs it after the download.
"""

import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(sys.argv[1])
    frame = pd.read_parquet(root / "raw/videomme/test-00000-of-00001.parquet")
    wanted = set(frame.loc[frame["duration"] == "long", "videoID"])
    out = root / "videos"
    out.mkdir(exist_ok=True)

    def extract(chunk: Path) -> int:
        count = 0
        with zipfile.ZipFile(chunk) as archive:
            for info in archive.infolist():
                name = Path(info.filename)
                if name.suffix == ".mp4" and name.stem in wanted:
                    target = out / name.name
                    with archive.open(info) as source, open(target,
                                                            "wb") as sink:
                        while block := source.read(1 << 24):
                            sink.write(block)
                    count += 1
        return count

    chunks = sorted((root / "raw").glob("videos_chunked_*.zip"))
    with ThreadPoolExecutor(len(chunks)) as pool:
        extracted = sum(pool.map(extract, chunks))
    missing = wanted - {path.stem for path in out.glob("*.mp4")}
    print(f"long videos: {len(wanted)}, extracted {extracted}, "
          f"missing {sorted(missing)}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

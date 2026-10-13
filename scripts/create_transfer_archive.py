"""Create an AES-256 ZIP of a portable EEG-XAI export."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import secrets

import pyzipper


def create_archive(source: Path, output: Path, password_file: Path, *, exclude_dataset: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    password_file = password_file.resolve()
    if not (source / "transfer_data" / "manifest.json").is_file():
        raise FileNotFoundError("A origem nao e um pacote EEG-XAI exportado.")
    if output.exists() or password_file.exists():
        raise FileExistsError("ZIP ou arquivo de senha ja existe; nao sera sobrescrito.")
    if source == output or source in output.parents or source in password_file.parents:
        raise ValueError("ZIP e senha devem ficar fora da pasta exportada.")
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and (not exclude_dataset or source / "dataset_amostra" not in path.parents)
    )
    password = secrets.token_urlsafe(32)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pyzipper.AESZipFile(
            output,
            mode="w",
            compression=pyzipper.ZIP_DEFLATED,
            compresslevel=3,
            allowZip64=True,
        ) as archive:
            archive.setpassword(password.encode("ascii"))
            archive.setencryption(pyzipper.WZ_AES, nbits=256)
            for index, path in enumerate(files, start=1):
                archive.write(path, arcname=f"{source.name}/{path.relative_to(source).as_posix()}")
                if index % 25 == 0 or index == len(files):
                    print(f"Arquivos compactados: {index}/{len(files)}", flush=True)
        with pyzipper.AESZipFile(output) as archive:
            archive.setpassword(password.encode("ascii"))
            bad_file = archive.testzip()
            if bad_file:
                raise RuntimeError(f"Falha na verificacao do ZIP: {bad_file}")
        digest = hashlib.sha256()
        with output.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        password_file.write_text(
            "Senha do ZIP EEG-XAI (nao envie este arquivo junto com o ZIP):\n"
            + password + "\n\nSHA-256 do ZIP:\n" + digest.hexdigest() + "\n",
            encoding="utf-8",
        )
    except Exception:
        if output.exists():
            output.unlink()
        raise
    print(f"ZIP verificado: {output} ({output.stat().st_size / (1024 ** 3):.2f} GiB)")
    print(f"Senha e SHA-256 salvos separadamente em: {password_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--exclude-dataset", action="store_true")
    args = parser.parse_args()
    create_archive(args.source, args.output, args.password_file, exclude_dataset=args.exclude_dataset)


if __name__ == "__main__":
    main()

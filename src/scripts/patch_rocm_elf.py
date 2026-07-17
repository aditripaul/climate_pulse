import contextlib
import glob
import struct


def fix_elf_stack(filename):
    with open(filename, "r+b") as f:
        f.seek(0)
        if f.read(4) != b"\x7fELF":
            return
        f.seek(4)
        if f.read(1)[0] != 2:
            return
        f.seek(32)
        e_phoff = struct.unpack("<Q", f.read(8))[0]
        f.seek(54)
        e_phentsize = struct.unpack("<H", f.read(2))[0]
        e_phnum = struct.unpack("<H", f.read(2))[0]
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            if struct.unpack("<I", f.read(4))[0] == 0x6474E551:
                p_flags = struct.unpack("<I", f.read(4))[0]
                if p_flags & 1:
                    print(f"Fixing executable stack on {filename}...")
                    f.seek(e_phoff + i * e_phentsize + 4)
                    f.write(struct.pack("<I", p_flags & ~1))
                return


for lib in glob.glob(".venv/lib/python*/site-packages/**/*.so", recursive=True):
    with contextlib.suppress(Exception):
        fix_elf_stack(lib)

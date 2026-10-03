import zipfile, os

wanted = ("NonVectrinoData/LISST/", "NonVectrinoData/CTD/")
wanted_p = [("NonVectrinoData/RBR/", "/P2/"), ("NonVectrinoData/ADP/", "/P1/")]

z = zipfile.ZipFile("NonVectrinoData.zip")
n = 0
for name in z.namelist():
    if name.endswith("/") or "__MACOSX" in name or "/._" in name:
        continue
    keep = name.startswith(wanted) or any(
        name.startswith(a) and b in name for a, b in wanted_p)
    if keep:
        z.extract(name, "data")
        n += 1
print("extracted", n, "files into ./data")
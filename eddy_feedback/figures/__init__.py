from string import ascii_lowercase

markers = {
    "CESM2": "oC3",
    "CMCC-CM2-SR5": "8C4",
    "CNRM-CM6-1": "^C0",
    "CNRM-ESM2-1": "vC0",
    "CanESM5": "sC5",
    "EC-Earth3": "pC6",
    "INM-CM5-0": "PC7",
    "IPSL-CM6A-LR": "*C8",
    "MIROC-ES2L": "^C1",
    "MIROC6": "vC1",
    "MPI-ESM1-2-HR": "^C2",
    "MPI-ESM1-2-LR": "vC2",
    "UKESM1-0-LL": "hC9",
}


def label_axes(axes, xpos=0.01, ypos=1.025):
    for n, ax in enumerate(axes.flatten()):
        ax.text(xpos, ypos, f"({ascii_lowercase[n]})", transform=ax.transAxes)

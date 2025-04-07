    for idx, docVector in enumerate(Dk):
        ax.text(docVector[0], docVector[1], docVector[2], testFiles[idx], size=10, zorder=1, color="k")
import matplotlib.pyplot as plt


def plot_nav(nav, title="NAV", output_path=None, show=False):
    ax = nav.plot(figsize=(10, 4))
    ax.set_title(title)
    ax.figure.tight_layout()
    if output_path is not None:
        ax.figure.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(ax.figure)

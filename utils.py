import textwrap
import matplotlib.pyplot as plt


def wrap_caption(text):
    # Calculate wrap width based on figure size
    fig = plt.gcf()
    fig_width_inches = fig.get_size_inches()[0]
    wrap_width = int(fig_width_inches * 10)

    return "\n".join(textwrap.wrap(text, width=wrap_width))

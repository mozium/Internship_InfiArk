import dash_html_components as html
from urllib.parse import quote as urlquote      # ***


def file_download_link(component, filename=None, id_='download-a-link'):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    if filename is None:
        return component
    location = "/models/{}".format(urlquote(filename))
    component.children = [
        html.A(filename, href=location, id=id_, hidden=True)
    ]
    return component


if __name__ == '__main__':
    pass
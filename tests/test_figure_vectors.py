import xml.etree.ElementTree as ET
import numpy as np


def test_visible_atomic_figure_names(tmp_path):
    from probemap.registration.figure_files import visible_figure_path,temporary_figure_path,publish_figure
    target=visible_figure_path(tmp_path/'..recording.pdf','.pdf')
    assert target.name=='recording.pdf'
    temporary=temporary_figure_path(target)
    assert not temporary.name.startswith('.')
    temporary.write_bytes(b'pdf')
    import os,stat
    if hasattr(os,'chflags') and hasattr(stat,'UF_HIDDEN'):
        os.chflags(temporary,temporary.stat().st_flags | stat.UF_HIDDEN)
    publish_figure(temporary,target)
    if hasattr(stat,'UF_HIDDEN'): assert not target.stat().st_flags & stat.UF_HIDDEN
    assert target.read_bytes()==b'pdf' and not temporary.exists()


def test_regions_are_separate_svg_paths_with_holes(tmp_path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from probemap.registration.figure_regions import matplotlib_region
    figure=Figure(); FigureCanvasAgg(figure); ax=figure.add_subplot()
    mask=np.zeros((12,12),bool); mask[1:11,1:11]=True; mask[4:8,4:8]=False
    one=matplotlib_region(ax,mask,[0,12,0,12],(1,0,0,.4),'region-1')
    matplotlib_region(ax,~mask,[0,12,0,12],(0,1,0,.4),'region-2')
    assert sum(one.get_path().codes==1)==2  # Separate outer and inner contours.
    path=tmp_path/'regions.svg'; figure.savefig(path)
    root=ET.parse(path).getroot(); ns={'s':'http://www.w3.org/2000/svg'}
    for identifier in ('region-1','region-2'):
        assert root.find(f".//s:g[@id='{identifier}']/s:path",ns) is not None
    assert not root.findall('.//s:image',ns)

package spawnn.gui;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;
import org.geotools.factory.CommonFactoryFinder;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.FeatureIterator;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.map.MapViewport;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.Mark;
import org.geotools.styling.SLD;
import org.geotools.styling.Stroke;
import org.geotools.styling.Style;
import org.geotools.styling.StyleBuilder;
import org.geotools.styling.Symbolizer;
import org.geotools.swing.JMapPane;
import org.geotools.swing.event.MapPaneEvent;
import org.geotools.swing.event.MapPaneListener;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.feature.type.GeometryType;
import org.opengis.filter.Filter;
import org.opengis.filter.FilterFactory2;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import net.miginfocom.swing.MigLayout;

public class MapPanel<T> extends NeuronVisPanel<T> implements MapPaneListener, ComponentListener, MouseListener {

	private static final long serialVersionUID = -8904376009449729813L;
	private static Logger log = Logger.getLogger(MapPanel.class);

	private FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2();
	private FeatureCollection<SimpleFeatureType, SimpleFeature> fc;
	private JMapPane mp;
	private List<T> pos;

	//TODO handle resize events properly
	public MapPanel(FeatureCollection<SimpleFeatureType, SimpleFeature> fc, List<T> pos) {
		super();

		this.pos = pos;
		this.fc = fc;
		setLayout(new MigLayout());

		mp = new JMapPane();
		mp.setDoubleBuffered(true);
		mp.setMapContent(new MapContent());

		GTRenderer renderer = new StreamingRenderer();
		RenderingHints hints = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		hints.put(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
		hints.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
		mp.setRenderer(renderer);

		mp.addMapPaneListener(this);
		mp.addComponentListener(this);
		mp.addMouseListener(this);
		mp.setOpaque(false); // FIXME this is ignored as soon as setColors is called
		add(mp, "push, grow");
	}

	@Override
	public void onDisplayAreaChanged(MapPaneEvent e) {
		mp.repaint(); //?
	}

	@Override
	public void onNewMapContent(MapPaneEvent arg0) {
	}

	@Override
	public void onRenderingStarted(MapPaneEvent arg0) {
	}

	@Override
	public void onRenderingStopped(MapPaneEvent arg0) {
	}

	@Override
	public void componentHidden(ComponentEvent arg0) {
	}

	@Override
	public void componentMoved(ComponentEvent arg0) {
	}

	@Override
	public void componentResized(ComponentEvent arg0) {
		mp.repaint(); //?
	}

	@Override
	public void componentShown(ComponentEvent arg0) {
	}

	private final int offset = (int) Math.ceil((1.0 + SELECTED_WIDTH) / 2);

	@Override
	public void saveImage(File fn, String mode) {
		Rectangle imageBounds = null;
		ReferencedEnvelope mapBounds = null;
		try {
			mapBounds = mp.getMapContent().getMaxBounds();
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			imageBounds = new Rectangle(offset, offset, mp.getWidth() + offset, (int) Math.round(mp.getWidth() * heightToWidth) + offset);

		} catch (Exception e) {
			throw new RuntimeException(e);

		}
		BufferedImage bufImage = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB);
		try {
			FileOutputStream stream = new FileOutputStream(fn);
			if (mode.equals("PNG")) {
				Graphics2D g = bufImage.createGraphics();
				g.drawImage(bufImage, 0, 0, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset, null);
				mp.getRenderer().paint(g, imageBounds, mapBounds);

				ImageIO.write(bufImage, "PNG", stream);
			} else if (mode.equals("EPS")) {
				EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
				g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
				g.setupDocument(stream, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset);
				mp.getRenderer().paint(g, imageBounds, mapBounds);
				g.finish();
			} else {
				log.debug("Unknown file format!");
			}
			stream.flush();
			stream.close();
		} catch (Exception e) {

		}
	}

	@Override
	public void setGridColors(Map<T, Color> colorMap, Map<T, Color> selectedColors, Map<T, Double> neuronValues) {
		StyleBuilder sb = new StyleBuilder();
		GeometryType gt = fc.getSchema().getGeometryDescriptor().getType();
		
		// set color attributes
		for (int i = 0; i < pos.size(); i++) {
			Color c = colorMap.get(pos.get(i));
			FeatureCollection<SimpleFeatureType, SimpleFeature> sub = fc.subCollection(ff.equals(ff.property("neuron"), ff.literal(i)));
			FeatureIterator<SimpleFeature> iter = sub.features();
			try {
				while (iter.hasNext())
					iter.next().setAttribute("color", c);
			} finally {
				iter.close();
			}
		}
		
		MapContent mc = new MapContent();
		ReferencedEnvelope bounds = new ReferencedEnvelope();		
				
		{
			Stroke stroke = sb.createStroke();
			// stroke = null; // draw border?
			Style style = null;
			if (gt.getBinding() == Polygon.class || gt.getBinding() == MultiPolygon.class) {
				Symbolizer sym = sb.createPolygonSymbolizer(stroke,sb.createFill(ff.property("color")));
				style = SLD.wrapSymbolizers(sym);
			} else if (gt.getBinding() == Point.class || gt.getBinding() == MultiPoint.class) {
				Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, sb.createFill(ff.property("color")), stroke);
				Symbolizer sym = sb.createPointSymbolizer(sb.createGraphic(null, mark, null));
				style = SLD.wrapSymbolizers(sym);
			} else {
				log.warn("GeomType not supported: " + gt.getBinding());
			}
			FeatureLayer fl = new FeatureLayer(fc, style);
			bounds.expandToInclude(fl.getBounds());
			mc.addLayer(fl);
		}
			
		// Maybe it would be better to have a single layer here
		List<Color> colors = new ArrayList<Color>();
		for( Color c : selectedColors.values() )
			if( !colors.contains(c) )
				colors.add(c);
		Collections.sort(colors,new Comparator<Color>() {
			@Override
			public int compare(Color o1, Color o2) {
				return Integer.compare(o1.getRGB(), o2.getRGB());
			}
		});
		Collections.reverse(colors);
		
		for( Color c : colors ) {
			// selected layer
			for (int i = 0; i < pos.size(); i++) {
				T t = pos.get(i);
				if (!selectedColors.containsKey(t) || selectedColors.get(t) != c )
					continue;
				
				FeatureCollection<SimpleFeatureType, SimpleFeature> sub = fc.subCollection(ff.equals(ff.property("neuron"), ff.literal(i)));
				if (selectSingle && sub.contains(selected)) {
					DefaultFeatureCollection dfc = new DefaultFeatureCollection();
					dfc.add(selected);
					sub = dfc;
				}

				Style style = null;
				if (gt.getBinding() == Polygon.class || gt.getBinding() == MultiPolygon.class) {
					style = SLD.wrapSymbolizers(sb.createPolygonSymbolizer(c, SELECTED_WIDTH));
				} else if (gt.getBinding() == Point.class || gt.getBinding() == MultiPoint.class) {
					Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, c, SELECTED_WIDTH);
					mark.setFill(null);
					style = SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)));
				} else {
					log.warn("GeomType not supported: " + gt.getBinding());
				}
				mc.addLayer(new FeatureLayer(sub, style));
			}
		}
				
		mc.setViewport( new MapViewport(bounds));			
		mp.setBackground(getBackground()); // quick and dirty hack. setOpaque(false) does not work somehow
		mp.getMapContent().dispose();
		mp.setMapContent(mc);
	}

	SimpleFeature selected = null;
	boolean selectSingle = false;

	@Override
	public void mouseClicked(MouseEvent evt) {
		AffineTransform af = mp.getScreenToWorldTransform();
		Point2D ptSrc = new Point2D.Double(evt.getPoint().getX(), evt.getPoint().getY());
		Point2D ptDst = null;
		ptDst = af.transform(ptSrc, ptDst);
		
		// find selected feature
		GeometryFactory gf = new GeometryFactory();
		//FIXME does not really work well for point features
		Geometry p = gf.createPoint(new Coordinate(ptDst.getX(), ptDst.getY()));
		Filter filter = ff.intersects(ff.property("the_geom"), ff.literal(p));
		try {
			FeatureCollection<SimpleFeatureType, SimpleFeature> sub = fc.subCollection(filter);
			FeatureIterator<SimpleFeature> iter = sub.features();
			try {
				while (iter.hasNext()) {
					SimpleFeature feature = iter.next();
					
					if (selectSingle)
						selected = feature;

					fireNeuronSelectedEvent(new NeuronSelectedEvent<T>(this, pos.get((Integer) feature.getAttribute("neuron"))));
				}
			} finally {
				iter.close();
			}
		} catch (Exception ex) {
			ex.printStackTrace();
			return;
		}
	}

	@Override
	public void mouseEntered(MouseEvent arg0) {
	}

	@Override
	public void mouseExited(MouseEvent arg0) {
	}

	@Override
	public void mousePressed(MouseEvent arg0) {
	}

	@Override
	public void mouseReleased(MouseEvent arg0) {
	}
}

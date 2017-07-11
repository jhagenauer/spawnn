package spawnn.gui;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.RenderingHints.Key;
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
import java.util.HashMap;
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
import org.geotools.map.Layer;
import org.geotools.map.MapContent;
import org.geotools.map.MapViewport;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.Fill;
import org.geotools.styling.Mark;
import org.geotools.styling.SLD;
import org.geotools.styling.Stroke;
import org.geotools.styling.Style;
import org.geotools.styling.StyleBuilder;
import org.geotools.styling.Symbolizer;
import org.geotools.swing.DefaultRenderingExecutor;
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

public class MapPanel<T> extends NeuronVisPanel<T> implements MapPaneListener, MouseListener, ComponentListener {

	private static final long serialVersionUID = -8904376009449729813L;
	private static Logger log = Logger.getLogger(MapPanel.class);

	private FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2();
	private StyleBuilder sb = new StyleBuilder();
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

		GTRenderer renderer = new StreamingRenderer();
		Map<Key, Object> hints = new HashMap<>();
		//hints.put(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		//hints.put(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
		hints.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
		renderer.setJava2DHints(new RenderingHints(hints));
		
		mp.setRenderer(renderer);
		mp.setRenderingExecutor(new DefaultRenderingExecutor());
				
		MapContent mc = new MapContent();
		mc.setViewport( new MapViewport(fc.getBounds() ) );		
		mp.setMapContent(mc);

		mp.addMapPaneListener(this);
		mp.addComponentListener(this);
		mp.addMouseListener(this);
		mp.setBackground(getBackground()); 
		mp.setOpaque(false); // FIXME this is ignored as soon as setColors is called
		add(mp, "push, grow");
	}

	private final int offset = (int) Math.ceil((1.0 + SELECTED_WIDTH) / 2);

	@Override
	public void saveImage(File fn, ImageMode mode) {
		Rectangle imageBounds = null;
		ReferencedEnvelope mapBounds = null;
		try {
			mapBounds = mp.getMapContent().getMaxBounds();
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			imageBounds = new Rectangle(offset, offset, mp.getWidth() + offset, (int) Math.round(mp.getWidth() * heightToWidth) + offset);

		} catch (Exception e) {
			throw new RuntimeException(e);

		}
		BufferedImage bufImage = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
		try {
			FileOutputStream stream = new FileOutputStream(fn);
			if (mode == ImageMode.PNG ) {
				Graphics2D g = bufImage.createGraphics();
				g.setComposite(AlphaComposite.Clear);
				int w = imageBounds.width + 2 * offset;
				int h = imageBounds.height + 2 * offset;
				g.fillRect(0, 0, w, h);
				g.setComposite(AlphaComposite.Src);
				g.drawImage(bufImage, 0, 0, w, h, null);
				mp.getRenderer().paint(g, imageBounds, mapBounds);

				ImageIO.write(bufImage, "PNG", stream);
			} else if (mode == ImageMode.EPS ) {
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
			e.printStackTrace();
		}
	}
	
	

	@Override
	public void setColors(Map<T, Color> colorMap, Map<T, Color> selectedColors, Map<T, Double> neuronValues) {
		GeometryType gt = fc.getSchema().getGeometryDescriptor().getType();
		List<Layer> layers = new ArrayList<>();
		{ // basic draw
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
			layers.add(new FeatureLayer(fc, style));
		}
				
		{ // select draw
			List<Filter> fl = new ArrayList<>();
			for (int i = 0; i < pos.size(); i++) {
				T t = pos.get(i);
				if (selectedColors.containsKey(t) )
					fl.add( ff.equals(ff.property("neuron"), ff.literal(i)));
			}
			
			FeatureCollection<SimpleFeatureType, SimpleFeature> sub = fc.subCollection(ff.or(fl));
			if (selectSingle && sub.contains(selected)) {
				DefaultFeatureCollection dfc = new DefaultFeatureCollection();
				dfc.add(selected);
				sub = dfc;
			}

			Style style = null;
			Stroke stroke = sb.createStroke(ff.property("selColor"),ff.literal(SELECTED_WIDTH));
			Fill fill = sb.createFill(ff.property("selColor"), ff.literal((double)NeuronVisPanel.SELECTED_OPACITY/256)); 
			if (gt.getBinding() == Polygon.class || gt.getBinding() == MultiPolygon.class) {
				Symbolizer sym = sb.createPolygonSymbolizer(stroke,fill);
				style = SLD.wrapSymbolizers(sym);
			} else if (gt.getBinding() == Point.class || gt.getBinding() == MultiPoint.class) {
				Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, fill, stroke);
				mark.setFill(null);
				style = SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)));
			} else {
				log.warn("GeomType not supported: " + gt.getBinding());
			}
			layers.add(new FeatureLayer(sub, style));	
		}
				
		MapContent mc = mp.getMapContent();	
		//mc.removeMapLayerListListener(mp);
		
		mc.layers().clear();
		mc.layers().addAll(layers);
		
		//mc.addMapLayerListListener(mp);
		//mc.fireLayerAdded(null, 0, layers.size()-1);
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
	public void mouseEntered(MouseEvent e) {
	}

	@Override
	public void mouseExited(MouseEvent e) {
	}

	@Override
	public void mousePressed(MouseEvent e) {
	}

	@Override
	public void mouseReleased(MouseEvent e) {
	}

	@Override
	public void componentHidden(ComponentEvent e) {
	}

	@Override
	public void componentMoved(ComponentEvent e) {
	}

	@Override
	public void componentShown(ComponentEvent e) {
	}
	
	@Override
	public void componentResized(ComponentEvent e) {
		//mp.reset();
		//mp.setSize(getSize());
		//mp.getMapContent().setViewport( new MapViewport(fc.getBounds()));
		//log.debug(mp.getSize()+":::"+getSize());
	}
	
	@Override
	public void onDisplayAreaChanged(MapPaneEvent e) {
	}

	@Override
	public void onNewMapContent(MapPaneEvent e) {
	}

	@Override
	public void onRenderingStarted(MapPaneEvent e) {
	}

	@Override
	public void onRenderingStopped(MapPaneEvent e) {
	}
}

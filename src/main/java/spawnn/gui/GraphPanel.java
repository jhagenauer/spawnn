package spawnn.gui;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Paint;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import org.apache.commons.collections15.Transformer;
import org.apache.commons.collections15.TransformerUtils;
import org.apache.log4j.Logger;
import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;

import edu.uci.ics.jung.algorithms.layout.AbstractLayout;
import edu.uci.ics.jung.algorithms.layout.CircleLayout;
import edu.uci.ics.jung.algorithms.layout.FRLayout;
import edu.uci.ics.jung.algorithms.layout.KKLayout;
import edu.uci.ics.jung.algorithms.layout.StaticLayout;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.visualization.Layer;
import edu.uci.ics.jung.visualization.VisualizationImageServer;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.control.DefaultModalGraphMouse;
import edu.uci.ics.jung.visualization.control.ModalGraphMouse;
import edu.uci.ics.jung.visualization.decorators.EdgeShape;
import edu.uci.ics.jung.visualization.picking.PickedState;
import edu.uci.ics.jung.visualization.transform.shape.GraphicsDecorator;
import edu.uci.ics.jung.visualization.util.Caching;

public class GraphPanel extends NeuronVisPanel<double[]> implements ItemListener, ComponentListener {

	private static Logger log = Logger.getLogger(GraphPanel.class);
	private static final long serialVersionUID = -7684883263291420601L;

	private Graph<double[], double[]> graph;
	private PickedState<double[]> pickedState;
	private VisualizationViewer<double[], double[]> vv;
	private int[] ga = null;

	// Edge weight layout modes
	public static String NONE = "None", DIST = "Distance", DIST_GEO = "Distance (geo)", COUNT = "Count";
	
	enum Layout {
		Circle, FruchteReingo, Geo, KamadaKawai,
	}
	
	Layout curLayout = Layout.KamadaKawai;
	

	GraphPanel( Graph<double[], double[]> graph, int[] ga, Layout curLayout ) {
		super();

		this.graph = graph;
		this.ga = ga;
		this.curLayout = curLayout;
		
		DefaultModalGraphMouse<double[], String> gm = new DefaultModalGraphMouse<double[], String>();
		gm.setMode(ModalGraphMouse.Mode.PICKING);
				
		vv = new VisualizationViewer<double[], double[]>(getGraphLayout());
		vv.getRenderContext().setEdgeShapeTransformer(new EdgeShape.Line<double[], double[]>());
		vv.setGraphMouse(gm);

		pickedState = vv.getPickedVertexState();
		pickedState.addItemListener(this);
		add(vv);
		
		addComponentListener(this);
	}
	
	@Override
	public void itemStateChanged(ItemEvent e) {
		Object subject = e.getItem();
		if (subject instanceof double[]) {
			double[] vertex = (double[]) subject;
			if (pickedState.isPicked(vertex)) {
				fireNeuronSelectedEvent(new NeuronSelectedEvent<double[]>(this, vertex));
				pickedState.clear();
			}
		}
	}

	@Override
	public void setColors(final Map<double[], Color> colorMap, final Map<double[], Color> selected, Map<double[], Double> neuronValues) {
		vv.getRenderContext().setVertexFillPaintTransformer(new Transformer<double[], Paint>() {
			public Paint transform(double[] i) {
				Color c = colorMap.get(i);
				if (selected.containsKey(i)) {
					Color s = selected.get(i);
					int alpha = NeuronVisPanel.SELECTED_OPACITY;
					int red = (s.getRed() * alpha + c.getRed() * (255 - alpha)) / 255;
					int green = (s.getGreen() * alpha + c.getGreen() * (255 - alpha)) / 255;
					int blue = (s.getBlue() * alpha + c.getBlue() * (255 - alpha)) / 255;
					return new Color(red, green, blue);
				}
				return c;
			}
		});

		vv.getRenderContext().setVertexDrawPaintTransformer(new Transformer<double[], Paint>() {
			public Paint transform(double[] i) {
				if (selected.containsKey(i))
					return selected.get(i);
				else
					return Color.BLACK;
			}
		});

		vv.getRenderContext().setVertexStrokeTransformer(new Transformer<double[], Stroke>() {
			@Override
			public Stroke transform(double[] i) {
				if (selected.containsKey(i))
					return new BasicStroke(SELECTED_WIDTH);
				else
					return new BasicStroke(1);
			}
		});				
		repaint();
		vv.repaint();
	}

	public void setEdgeStyle(final String mode) {
		double max = Double.NEGATIVE_INFINITY, maxGeo = Double.NEGATIVE_INFINITY, maxCount = Double.NEGATIVE_INFINITY;
		for (double[] edge : graph.getEdges()) {
			maxCount = Math.max(maxCount, edge[0]);
			max = Math.max(max, edge[1]);
			if (edge.length > 2)
				maxGeo = Math.max(maxGeo, edge[2]);
		}

		final double MAX = max, MAX_GEO = maxGeo, MAX_COUNT = maxCount;

		vv.getRenderContext().setEdgeStrokeTransformer(new Transformer<double[], Stroke>() {
			@Override
			public Stroke transform(double[] edge) {
				double width = 0.1; // default
				if (mode == GraphPanel.COUNT)
					width = edge[0] / MAX_COUNT;
				else if (mode == GraphPanel.DIST)
					width = 1.0 - edge[1] / MAX;
				else if (mode == GraphPanel.DIST_GEO && edge.length > 2)
					width = 1.0 - edge[2] / MAX_GEO;
				return new BasicStroke(10.0f * (float) width);
			}
		});

		repaint();
		vv.repaint();
	}
	
	public void setGraphLayout( Layout lm ) {
		this.curLayout = lm;
		
		// update
		vv.setSize(getSize()); // necessary, because getGraphLayout uses vv.getSize()
		AbstractLayout<double[],double[]> al = getGraphLayout();		
		vv.setGraphLayout(al);
	}

	private AbstractLayout<double[],double[]> getGraphLayout() {
		AbstractLayout<double[],double[]> al = null;
		if (curLayout == Layout.KamadaKawai) {
			al = new KKLayout<double[], double[]>(graph);
		} else if (curLayout == Layout.Circle) {
			al = new CircleLayout<double[], double[]>(graph);
		} else if (curLayout == Layout.FruchteReingo) {
			al = new FRLayout<double[], double[]>(graph);
		} else if (curLayout == Layout.Geo) {
			al = new GeoLayout<>(graph, ga);
		} 
		return al;
	}

	@Override
	public void saveImage(File fn, ImageMode mode) {
		
		// all this inherited code just to allow transparent background
		VisualizationImageServer<double[], double[]> vis = new VisualizationImageServer<double[], double[]>(vv.getGraphLayout(), vv.getGraphLayout().getSize()) {
			private static final long serialVersionUID = -808579299453410392L;

			@Override
			protected void renderGraph(Graphics2D g2d) {
				if (renderContext.getGraphicsContext() == null) {
					renderContext.setGraphicsContext(new GraphicsDecorator(g2d));
				} else {
					renderContext.getGraphicsContext().setDelegate(g2d);
				}
				renderContext.setScreenDevice(this);
				edu.uci.ics.jung.algorithms.layout.Layout<double[], double[]> layout = model.getGraphLayout();

				Map<RenderingHints.Key, Object> renderingHints = new HashMap<RenderingHints.Key, Object>();
				renderingHints.put(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				g2d.setRenderingHints(renderingHints);

				Dimension d = getSize();
				g2d.setComposite(AlphaComposite.Clear);
				g2d.fillRect(0, 0, d.width, d.height);
				g2d.setComposite(AlphaComposite.Src);

				AffineTransform oldXform = g2d.getTransform();
				AffineTransform newXform = new AffineTransform(oldXform);
				newXform.concatenate(renderContext.getMultiLayerTransformer().getTransformer(Layer.VIEW).getTransform());
				g2d.setTransform(newXform);

				for (Paintable paintable : preRenderers) {
					if (paintable.useTransform()) {
						paintable.paint(g2d);
					} else {
						g2d.setTransform(oldXform);
						paintable.paint(g2d);
						g2d.setTransform(newXform);
					}
				}

				if (layout instanceof Caching)
					((Caching) layout).clear();

				renderer.render(renderContext, layout);

				for (Paintable paintable : postRenderers) {
					if (paintable.useTransform()) {
						paintable.paint(g2d);
					} else {
						g2d.setTransform(oldXform);
						paintable.paint(g2d);
						g2d.setTransform(newXform);
					}
				}
				g2d.setTransform(oldXform);
			}

			@Override
			public Image getImage(Point2D center, Dimension d) {
				int width = getWidth();
				int height = getHeight();

				float scalex = (float) width / d.width;
				float scaley = (float) height / d.height;
				try {
					renderContext.getMultiLayerTransformer().getTransformer(Layer.VIEW).scale(scalex, scaley, center);
					BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
					Graphics2D graphics = bi.createGraphics();
					Map<RenderingHints.Key, Object> renderingHints = new HashMap<RenderingHints.Key, Object>();
					renderingHints.put(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
					graphics.setRenderingHints(renderingHints);
					paint(graphics);
					graphics.dispose();
					return bi;
				} finally {
					renderContext.getMultiLayerTransformer().getTransformer(Layer.VIEW).setToIdentity();
				}
			}
		};
		// VisualizationImageServer<double[], double[]> vis = new VisualizationImageServer<double[], double[]>(vv.getGraphLayout(), vv.getGraphLayout().getSize());
		// vis.setBackground(Color.WHITE);

		vis.getRenderContext().setVertexFillPaintTransformer(vv.getRenderContext().getVertexFillPaintTransformer());
		vis.getRenderContext().setVertexStrokeTransformer(vv.getRenderContext().getVertexStrokeTransformer());
		vis.getRenderContext().setVertexDrawPaintTransformer(vv.getRenderContext().getVertexDrawPaintTransformer());

		vis.getRenderContext().setEdgeShapeTransformer(vv.getRenderContext().getEdgeShapeTransformer());
		vis.getRenderContext().setEdgeStrokeTransformer(vv.getRenderContext().getEdgeStrokeTransformer());

		double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
		double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
		AbstractLayout<double[],double[]> al = getGraphLayout();
		for (double[] d : graph.getVertices()) {
			minX = Math.min(minX, al.getX(d));
			minY = Math.min(minY, al.getY(d));
			maxX = Math.max(maxX, al.getX(d));
			maxY = Math.max(maxY, al.getY(d));
		}

		double w = vv.getGraphLayout().getSize().getWidth();
		double h = vv.getGraphLayout().getSize().getHeight();
		Point2D.Double center = new Point2D.Double(w / 2, h / 2);
		BufferedImage bufImage = (BufferedImage) vis.getImage(center, new Dimension(vv.getGraphLayout().getSize()));

		try {
			FileOutputStream stream = new FileOutputStream(fn);
			if (mode == ImageMode.PNG ) {
				//Graphics2D g = bufImage.createGraphics();
				//g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				ImageIO.write(bufImage, "PNG", stream);
			} else if (mode == ImageMode.EPS ) {
				EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
				g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
				g.setupDocument(stream, bufImage.getWidth(), bufImage.getHeight());
				vis.print(g);
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
	public void componentResized(ComponentEvent e) {
		vv.setSize(getSize()); // necessary, because getGraphLayout uses vv.getSize()
		AbstractLayout<double[],double[]> al = getGraphLayout();		
		vv.setGraphLayout(al);
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
}

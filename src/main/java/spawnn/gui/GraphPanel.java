package spawnn.gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Paint;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
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
import edu.uci.ics.jung.visualization.VisualizationImageServer;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.control.DefaultModalGraphMouse;
import edu.uci.ics.jung.visualization.control.ModalGraphMouse;
import edu.uci.ics.jung.visualization.decorators.EdgeShape;
import edu.uci.ics.jung.visualization.picking.PickedState;

public class GraphPanel extends NeuronVisPanel<double[]> implements ItemListener {

	private static Logger log = Logger.getLogger(GraphPanel.class);
	private static final long serialVersionUID = -7684883263291420601L;

	private Graph<double[], double[]> graph;
	private PickedState<double[]> ps;
	private VisualizationViewer<double[], double[]> vv;
	private int[] ga = null;

	enum Layout { Circle, FruchteReingo, Geo, KamadaKawai }
	
	// Edge weight layout modes
	public static String NONE = "None", DIST = "Distance", DIST_GEO = "Distance (geo)", COUNT = "Count";

	AbstractLayout<double[], double[]> al = null;
	
	GraphPanel(Graph<double[], double[]> graph, int[] ga) {
		super();

		this.graph = graph;
		this.ga = ga;

		al = new KKLayout<double[], double[]>(graph);
		vv = new VisualizationViewer<double[], double[]>(al);
		vv.getRenderContext().setEdgeShapeTransformer(new EdgeShape.Line<double[], double[]>());
		
		DefaultModalGraphMouse<double[], String> gm = new DefaultModalGraphMouse<double[], String>();
		gm.setMode(ModalGraphMouse.Mode.PICKING);

		vv.setGraphMouse(gm);

		ps = vv.getPickedVertexState();
		ps.addItemListener(this);
		
		add(vv);			
	}

	@Override
	public void itemStateChanged(ItemEvent e) {
		Object subject = e.getItem();
		if (subject instanceof double[]) {
			double[] vertex = (double[]) subject;
			if (ps.isPicked(vertex)) {
				fireNeuronSelectedEvent(new NeuronSelectedEvent<double[]>(this, vertex));
				ps.clear();
			}
		}
	}

	@Override
	public void setGridColors(final Map<double[], Color> colorMap, final Map<double[], Color> selected, Map<double[],Double> neuronValues ) {
		vv.getRenderContext().setVertexFillPaintTransformer(new Transformer<double[], Paint>() {
			public Paint transform(double[] i) {
				if (colorMap.containsKey(i))
					return colorMap.get(i);
				else
					return Color.GRAY;
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
		
		vv.getRenderContext().setVertexStrokeTransformer( new Transformer<double[], Stroke>() {
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
	
	public void setEdgeStyle( final String mode ) {
		double max = Double.NEGATIVE_INFINITY, maxGeo = Double.NEGATIVE_INFINITY, maxCount = Double.NEGATIVE_INFINITY;
		for (double[] edge : graph.getEdges()) {
			maxCount = Math.max(maxCount, edge[0]);
			max = Math.max(max, edge[1]);
			if( edge.length > 2 )
				maxGeo = Math.max(maxGeo, edge[2]);
		}
		
		final double MAX = max, MAX_GEO = maxGeo, MAX_COUNT = maxCount;
		
		vv.getRenderContext().setEdgeStrokeTransformer( new Transformer<double[], Stroke>() {
			@Override
			public Stroke transform(double[] edge) {
				double width = 0.1; // default
				if( mode == GraphPanel.COUNT )
					width = edge[0]/MAX_COUNT;
				else if (mode == GraphPanel.DIST)
					width = 1.0 - edge[1] / MAX;
				else if (mode == GraphPanel.DIST_GEO && edge.length > 2 )
					width = 1.0 - edge[2] / MAX_GEO;
				return new BasicStroke( 10.0f*(float)width );
			}
		});
		
		repaint();
		vv.repaint();
	}

	public void setGraphLayout(Layout lm) {
		if (lm == Layout.KamadaKawai) {
			al = new KKLayout<double[], double[]>(graph);
		} else if (lm == Layout.Circle) {
			al = new CircleLayout<double[], double[]>(graph);
		} else if (lm == Layout.FruchteReingo) {
			al = new FRLayout<double[], double[]>(graph);
		} else if (lm == Layout.Geo) {
			int border = 12; // TODO this can be done more elegant
			Dimension dim = vv.getSize();

			double minX = Double.POSITIVE_INFINITY, minY = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
			for (double[] d : graph.getVertices()) {
				minX = Math.min(minX, d[ga[0]]);
				maxX = Math.max(maxX, d[ga[0]]);
				minY = Math.min(minY, -d[ga[1]]);
				maxY = Math.max(maxY, -d[ga[1]]);
			}

			Map<double[], Point2D> map = new HashMap<double[], Point2D>();
			for (double[] d : graph.getVertices()) {
				// keep aspect ratio
				double s1 = Math.max(maxX - minX, maxY - minY);
				double s2 = Math.max(dim.getWidth(), dim.getHeight());
				map.put(d, new Point2D.Double((s2 - 2 * border) * (d[ga[0]] - minX) / s1 + border, (s2 - 2 * border) * (-d[ga[1]] - minY) / s1 + border));
			}

			Transformer<double[], Point2D> vertexLocations = TransformerUtils.mapTransformer(map);
			al = new StaticLayout<double[], double[]>(graph, vertexLocations);
		}
		vv.setGraphLayout(al);
	}
	
	@Override
	public void saveImage( File fn, String mode ) {
		VisualizationImageServer<double[], double[]> vis = new VisualizationImageServer<double[], double[]>(vv.getGraphLayout(), vv.getGraphLayout().getSize());
		vis.setBackground(Color.WHITE);
		
		vis.getRenderContext().setVertexFillPaintTransformer(vv.getRenderContext().getVertexFillPaintTransformer());
		vis.getRenderContext().setVertexStrokeTransformer( vv.getRenderContext().getVertexStrokeTransformer());
		vis.getRenderContext().setVertexDrawPaintTransformer( vv.getRenderContext().getVertexDrawPaintTransformer());
				
		vis.getRenderContext().setEdgeShapeTransformer(vv.getRenderContext().getEdgeShapeTransformer());
		vis.getRenderContext().setEdgeStrokeTransformer(vv.getRenderContext().getEdgeStrokeTransformer());

		double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
		double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
		for( double[] d : graph.getVertices() ) {
			minX = Math.min(minX,al.getX(d));
			minY = Math.min(minY,al.getY(d));
			maxX = Math.max(maxX,al.getX(d));
			maxY = Math.max(maxY,al.getY(d));
		}
		
		double w = vv.getGraphLayout().getSize().getWidth();
		double h = vv.getGraphLayout().getSize().getHeight();
		Point2D.Double center = new Point2D.Double(w / 2, h / 2);
		
		/*float scale = (float)Math.min( 
				Math.min( 300.0/(300-minX-25) , 300.0/(600-maxX-25) ), 
				Math.min( 300.0/(300-minY-25) , 300.0/(600-maxY-25) ) );
		ScalingControl vvsc = new CrossoverScalingControl();
		vvsc.scale(vis, scale, center );*/
		//vis.getGraphLayout().setSize( new Dimension( (int)(maxX-minX) , (int)(maxY-minY) ) );
				
		BufferedImage bufImage = (BufferedImage) vis.getImage(center,new Dimension(vv.getGraphLayout().getSize()));
				
		try {		
			FileOutputStream stream = new FileOutputStream(fn);			
			if( mode == "PNG" ) {
				Graphics2D g = bufImage.createGraphics();
				g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				ImageIO.write(bufImage, "PNG", stream);
			} else if( mode == "EPS") {
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
		} catch( Exception e ) {
			
		}
	}
}

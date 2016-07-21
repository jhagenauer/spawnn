package spawnn.gui;

import java.awt.Dimension;
import java.awt.geom.Point2D;

import edu.uci.ics.jung.algorithms.layout.AbstractLayout;
import edu.uci.ics.jung.graph.Graph;

public class GeoLayout <E> extends AbstractLayout<double[], E> {

	private int ga[];
	private double minX = Double.POSITIVE_INFINITY, minY = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;

	public GeoLayout(Graph<double[], E> graph, int[] ga) {
		super(graph);
		this.ga = ga;
		
		for (double[] d : graph.getVertices()) {
			minX = Math.min(minX, d[ga[0]]);
			maxX = Math.max(maxX, d[ga[0]]);
			minY = Math.min(minY, -d[ga[1]]);
			maxY = Math.max(maxY, -d[ga[1]]);
		}
	}

	@Override
	public void initialize() {
		Dimension dim = getSize();
		double margin = 16;

		if (dim != null) {	
			double s1 , s2;
			if( maxX - minX > maxY - minY ) // geo width > height
				s1 = maxX - minX;
			else
				s1 = maxY - minY;
			
			if( dim.getWidth() > dim.getHeight() )
				s2 = dim.getWidth();
			else
				s2 = dim.getHeight();
			
			s2 -= 2*margin;

			for (double[] d : graph.getVertices() ) {
				Point2D coord = transform(d);
				coord.setLocation(
						s2 * ( d[ga[0]] - minX) / s1 + margin, 
						s2 * (-d[ga[1]] - minY) / s1 + margin
						);
			}
		}
	}

	@Override
	public void reset() {
		initialize();
	}
}

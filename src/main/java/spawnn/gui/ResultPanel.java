package spawnn.gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.imageio.ImageIO;
import javax.swing.JPanel;

import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;

import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

//TODO Probably a lot of code from the result panels can/should be placed here
public abstract class ResultPanel<T> extends JPanel {

	private static final long serialVersionUID = 1686748469941486349L;

	public static Map<double[], Set<double[]>> prototypeClusterToDataCluster(Map<double[], Set<double[]>> nBmus, List<Set<double[]>> clusters) {
		Map<double[], Set<double[]>> ll = new HashMap<double[], Set<double[]>>();
		for (Set<double[]> s : clusters) {
			Set<double[]> l = new HashSet<double[]>();
			for (double[] p : s)
				if (nBmus.containsKey(p))
					l.addAll(nBmus.get(p));
			if (!l.isEmpty())
				ll.put(DataUtils.getMeanClusterElement(l), l);
		}
		return ll;
	}

	public FeatureCollection<SimpleFeatureType, SimpleFeature> buildClusterFeatures(SpatialDataFrame sd, List<double[]> samples, Map<T, Set<double[]>> cluster, List<T> pos) {
		SimpleFeatureTypeBuilder sftb = new SimpleFeatureTypeBuilder();
		sftb.setName("data");
		sftb.setCRS(sd.crs);
		for (int i = 0; i < sd.names.size(); i++) {
			if (sd.bindings.get(i) == SpatialDataFrame.binding.Double)
				sftb.add(sd.names.get(i), Double.class);
			else if (sd.bindings.get(i) == SpatialDataFrame.binding.Integer)
				sftb.add(sd.names.get(i), Integer.class);
			else if (sd.bindings.get(i) == SpatialDataFrame.binding.Long)
				sftb.add(sd.names.get(i), Long.class);
		}
		sftb.add("neuron", Integer.class);
		sftb.add("nValue", Double.class);
		sftb.add("selected", Integer.class);

		Geometry g = sd.geoms.get(0);
		if (g instanceof Polygon)
			sftb.add("the_geom", Polygon.class);
		else if (g instanceof MultiPolygon)
			sftb.add("the_geom", MultiPolygon.class);
		else if (g instanceof Point)
			sftb.add("the_geom", Point.class);
		else if (g instanceof MultiPoint)
			sftb.add("the_geom", MultiPoint.class);
		else
			throw new RuntimeException("Unkown geometry type!");

		SimpleFeatureType type = sftb.buildFeatureType();
		SimpleFeatureBuilder builder = new SimpleFeatureBuilder(type);

		DefaultFeatureCollection fc = new DefaultFeatureCollection();
		for (int k = 0; k < samples.size(); k++) {
			double[] d = samples.get(k);
			for (int i = 0; i < pos.size(); i++) {
				T p = pos.get(i);
				if (cluster.containsKey(p) && cluster.get(p).contains(d)) {
					for (int j = 0; j < sd.names.size(); j++)
						builder.set(sd.names.get(j), sd.samples.get(k)[j]);
					builder.set("neuron", i);
					builder.set("the_geom", sd.geoms.get(k));
					fc.add(builder.buildFeature(fc.size() + ""));
					break;
				}
			}
		}
		return fc;
	}

	private static <T> void drawLegend(Graphics2D g, final Map<T, Color> colorMap, final Map<T, Double> neuronValues, int width, int height, boolean outline) {
		Map<Double, Color> cols = new HashMap<Double, Color>();
		for (T neuron : colorMap.keySet())
			cols.put(neuronValues.get(neuron), colorMap.get(neuron));
		final List<Double> sortedKeys = new ArrayList<Double>(cols.keySet());
		Collections.sort(sortedKeys);

		final int symbolWidth = 16;
		final int ySpacing = 18;
		final int xSpacing = 100;
		final int symbolHeight = 11;
		final int fontSize = 14;
		g.setStroke(new BasicStroke(1.0f)); // necessary because of bug of geotools
		g.setFont(new Font(g.getFont().getFontName(), Font.PLAIN, fontSize));

		int xOffset = 2;
		int yOffset = 2;

		int x = xOffset;
		int y = yOffset;
		int i = 0;
		for (Double d : sortedKeys) {
			if (x > width) { // new row
				x = xOffset;
				y += ySpacing;
			}

			g.setColor(cols.get(d));
			g.fillRect(x, y, symbolWidth, symbolHeight);

			g.setColor(Color.BLACK);

			if (outline)
				g.drawRect(x, y, symbolWidth, symbolHeight);

			g.drawString("Cluster " + (i + 1), x + 22, y + symbolHeight);

			i++;
			x += xSpacing;

		}
	}

	private static <T> void drawContLegend(Graphics2D g, final Map<T, Color> colorMap, final Map<T, Double> neuronValues, int width, int height) {
		Map<Double, Color> cols = new HashMap<Double, Color>();
		for (T neuron : colorMap.keySet())
			cols.put(neuronValues.get(neuron), colorMap.get(neuron));
		final List<Double> sortedKeys = new ArrayList<Double>(cols.keySet());
		Collections.sort(sortedKeys);

		final int symbolHeight = 11;
		final int fontSize = 14;
		g.setStroke(new BasicStroke(1.0f)); // necessary because of bug of geotools
		g.setFont(new Font(g.getFont().getFontName(), Font.PLAIN, fontSize));

		int xOffset = 2;
		int yOffset = 4;
		double x = xOffset;
		int y = yOffset;

		int scaleWidth = width - 40;
		double colWidth = (double) scaleWidth / sortedKeys.size();

		DecimalFormat df = new DecimalFormat("#0.00");
		for (Double d : sortedKeys) {
			g.setColor(Color.BLACK);
			if (d == sortedKeys.get(0) || d == sortedKeys.get(sortedKeys.size() - 1)) {
				g.drawLine((int) Math.floor(fontSize + x + 0.5 * colWidth), y, (int) Math.floor(fontSize + x + 0.5 * colWidth), (int) Math.round(y + 1.25 * symbolHeight));
				g.drawString(df.format(d.doubleValue()).replace(',', '.'), (int) Math.round(x), (int) (Math.round(y + 2.35 * symbolHeight)));

			}

			g.setColor(cols.get(d));
			g.fillRect((int) Math.floor(x + fontSize), y, (int) Math.ceil(colWidth), symbolHeight);

			x += colWidth;
		}

		g.setColor(Color.BLACK);
		g.drawRect(xOffset + fontSize, yOffset, scaleWidth, symbolHeight);
	}

	public static <T> void saveLegend(final Map<T, Color> colorMap, final Map<T, Double> neuronValues, File fn, String fileMode) {
		saveLegend(colorMap, neuronValues, fn, fileMode, true, (int) Math.ceil((double) ((new HashSet<Color>(colorMap.values())).size()) / 2));
	}

	// TODO Buggy, this is not for productive means
	public static <T> void saveLegend(final Map<T, Color> colorMap, final Map<T, Double> neuronValues, File fn, String fileMode, boolean outline, int nrCols) {
		try {
			int numColors = (new HashSet<Color>(colorMap.values())).size();

			FileOutputStream stream = new FileOutputStream(fn);
			if (fileMode == "PNG") {
				if (numColors > 12) { // cont legend
					int width = 390, height = 35;
					BufferedImage bufImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
					Graphics2D g = bufImage.createGraphics();
					g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
					drawContLegend(g, colorMap, neuronValues, width, height);
					ImageIO.write(bufImage, "PNG", stream);
				} else { // small legend
					int width = 600, height = 35; // each col = 100
					BufferedImage bufImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
					Graphics2D g = bufImage.createGraphics();
					g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
					drawLegend(g, colorMap, neuronValues, width, height, outline);
					ImageIO.write(bufImage, "PNG", stream);
				}
			} else if (fileMode == "EPS") {
				if (numColors > 12) { // cont legend
					int width = 390, height = 35;
					EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
					g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
					g.setupDocument(stream, width, height);
					drawContLegend(g, colorMap, neuronValues, width, height);
					g.finish();
				} else {
					int width = nrCols * 100, height = 35; // each col = 100
					EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
					g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
					g.setupDocument(stream, width, height);
					drawLegend(g, colorMap, neuronValues, width, height, outline);
					g.finish();
				}

			} else {
				System.out.println("Unknown file format!");
			}
			stream.flush();
			stream.close();
		} catch (Exception e) {

		}
	}

	public static <T> void saveLegend2(final Map<T, Color> colorMap, final Map<T, Double> neuronValues, File fn, boolean outline, boolean ng ) {
		try {
			FileOutputStream stream = new FileOutputStream(fn);

			int width = 596, height = 35; // ??? what is best width?
			if( ng )
				width = 662;
			EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
			g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
			g.setupDocument(stream, width, height);
					
			Map<Double, Color> cols = new HashMap<Double, Color>();
			for (T neuron : colorMap.keySet())
				cols.put(neuronValues.get(neuron), colorMap.get(neuron));
			final List<Double> sortedKeys = new ArrayList<Double>(cols.keySet());
			Collections.sort(sortedKeys);

			final int symbolWidth = 16;
			final int ySpacing = 18;
			final int symbolHeight = 11;
			final int fontSize = 14;
			g.setStroke(new BasicStroke(1.0f)); // necessary because of bug of geotools
			g.setFont(new Font(g.getFont().getFontName(), Font.PLAIN, fontSize));

			int xOffset = 2;
			int yOffset = 2;

			Map<Integer,Double> ws = new HashMap<Integer,Double>();
			int x = xOffset;
			int y = yOffset;
			int i = 0;
			for (Double d : sortedKeys) {
				if (i == 4 || x > width ) { // new row
					x = xOffset;
					y += ySpacing;
				}

				g.setColor(cols.get(d));
				g.fillRect(x, y, symbolWidth, symbolHeight);

				g.setColor(Color.BLACK);

				if (outline)
					g.drawRect(x, y, symbolWidth, symbolHeight);
				x += symbolWidth + 4;
				
				int clusterNr = i+1;
				String clusterDesc = "";
				int extraSpace = 0;
				
				if( clusterNr == 1 ) {
					clusterDesc = " (African Americans)";
					extraSpace = 10;
				} else if( clusterNr == 3 ) {
					clusterDesc = " (Universities)";
					extraSpace = 8;
				} else if( clusterNr == 5 ) {
					clusterDesc = " (Hispanics)";
					extraSpace = 8;
				} else if( clusterNr == 4 && ng ) { 
					clusterDesc = " (Asians)";
					extraSpace = 5;
				} else if( clusterNr == 8 && ng ) {
					clusterDesc =" (Seniors)";
					extraSpace = 6;
				} else {
					extraSpace = 3;
				}
				extraSpace += 4;
				
				String st = "Cluster " + clusterNr + clusterDesc;
				g.drawString(st, x, y + symbolHeight);
				
				double w = g.getFontMetrics().stringWidth(st) + extraSpace;
				if( i < 4 )
					ws.put(i, w);
				else 
					w = ws.get( i-4 );
				x += w; 
				i++;
			}
						
			g.finish();

			stream.flush();
			stream.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}

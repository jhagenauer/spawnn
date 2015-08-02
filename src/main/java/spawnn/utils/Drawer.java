package spawnn.utils;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.map.MapViewport;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.Mark;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.geotools.swing.JMapPane;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

public class Drawer {

	private static Logger log = Logger.getLogger(Drawer.class);

	private static Color[] col;

	static {
		Random r = new Random();
		Color[] c = new Color[] { Color.GREEN, Color.RED, Color.BLUE, Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.ORANGE, Color.PINK, Color.GRAY };
		col = new Color[c.length + 10];
		for (int i = 0; i < c.length; i++)
			col[i] = c[i];
		for (int i = c.length; i < col.length; i++)
			col[i] = new Color(r.nextInt());
	}

	// get temperature color in the range 0...255f
	public static Color getColor(float v) {
		float f = 1 - (v * 256.0f / 360) + 256f / 360;
		return new Color(Color.HSBtoRGB(f, 1f, 1f));
	}

	public static Color getColor(int i) {
		return col[i];
	}

	public static void geoDrawCluster(Collection<Set<double[]>> cluster, List<double[]> samples, List<Geometry> geoms, String fn, boolean label) {
		try {
			geoDrawCluster(cluster, samples, geoms, new FileOutputStream(fn), label);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void geoDrawCluster(Collection<Set<double[]>> cluster, List<double[]> samples, List<Geometry> geoms, OutputStream os, boolean label) {

		int nonEmpty = 0;
		Map<double[], Double> valueMap = new HashMap<double[], Double>();
		for (Collection<double[]> l : cluster) {
			if (l.isEmpty())
				continue;

			for (double[] d : l)
				valueMap.put(d, (double) nonEmpty);
			nonEmpty++;
		}

		Map<double[], Color> colorMap = ColorBrewerUtil.valuesToColors(valueMap, ColorBrewerUtil.ColorMode.Set3);

		// draw
		try {
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("cluster");
			typeBuilder.add("cluster", Integer.class);

			if (geoms.get(0) instanceof Polygon)
				typeBuilder.add("the_geom", Polygon.class);
			else if (geoms.get(0) instanceof Point)
				typeBuilder.add("the_geom", Point.class);
			else if (geoms.get(0) instanceof MultiPolygon)
				typeBuilder.add("the_geom", MultiPolygon.class);
			else
				throw new RuntimeException("Unknown Geometry type: " + geoms.get(0));

			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());

			StyleBuilder sb = new StyleBuilder();
			MapContent map = new MapContent();
			ReferencedEnvelope maxBounds = null;

			int clusterIndex = 0;
			for (Collection<double[]> l : cluster) {
				if (l.isEmpty())
					continue;

				DefaultFeatureCollection fc = new DefaultFeatureCollection();
				for (double[] d : l) {
					int idx = samples.indexOf(d);
					featureBuilder.add(clusterIndex);
					featureBuilder.add(geoms.get(idx));
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}

				if (maxBounds == null)
					maxBounds = fc.getBounds();
				else
					maxBounds.expandToInclude(fc.getBounds());

				double[] first = l.iterator().next();
				Color color = colorMap.get(first);

				if (geoms.get(0) instanceof Polygon || geoms.get(0) instanceof MultiPolygon)
					map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(color, Color.BLACK, 1.0))));
				else if (geoms.get(0) instanceof Point) {
					Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, color);
					map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)))));
				} else
					throw new RuntimeException("No layer for geometry type added");

				if (label)
					map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createTextSymbolizer(Color.BLACK, sb.createFont("Arial", 10), "cluster"))));
				clusterIndex++;
			}

			GTRenderer renderer = new StreamingRenderer();
			RenderingHints hints = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			hints.put(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
			hints.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
			//renderer.setRendererHints(hints);
			renderer.setMapContent(map);

			Rectangle imageBounds = null;
			try {
				double heightToWidth = maxBounds.getSpan(1) / maxBounds.getSpan(0);
				int imageWidth = 1024;

				imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
				// imageBounds = new Rectangle( 0, 0, mp.getWidth(), (int) Math.round(mp.getWidth() * heightToWidth));

				BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB);

				// png
				Graphics2D gr = image.createGraphics();
				gr.setPaint(Color.WHITE);
				gr.fill(imageBounds);
				
				renderer.paint(gr, imageBounds, maxBounds);

				ImageIO.write(image, "png", os);
				image.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			map.dispose();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void geoDrawClusterEPS(Collection<Set<double[]>> cluster, List<double[]> samples, List<Geometry> geoms, String fn, boolean border) {

		int offset = (int) Math.ceil((1.0 + 3) / 2);
		int nonEmpty = 0;
		Map<double[], Double> valueMap = new HashMap<double[], Double>();
		for (Collection<double[]> l : cluster) {
			if (l.isEmpty())
				continue;

			for (double[] d : l)
				valueMap.put(d, (double) nonEmpty);
			nonEmpty++;
		}

		Map<double[], Color> colorMap = ColorBrewerUtil.valuesToColors(valueMap, ColorBrewerUtil.ColorMode.Set3);

		// draw
		try {
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("cluster");
			typeBuilder.add("cluster", Integer.class);

			if (geoms.get(0) instanceof Polygon)
				typeBuilder.add("the_geom", Polygon.class);
			else if (geoms.get(0) instanceof Point)
				typeBuilder.add("the_geom", Point.class);
			else if (geoms.get(0) instanceof MultiPolygon)
				typeBuilder.add("the_geom", MultiPolygon.class);
			else
				throw new RuntimeException("Unknown Geometry type: " + geoms.get(0));

			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());

			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();
			ReferencedEnvelope maxBounds = null;
			
			int clusterIndex = 0;
			for (Collection<double[]> l : cluster) {
				if (l.isEmpty())
					continue;

				DefaultFeatureCollection fc = new DefaultFeatureCollection();
				for (double[] d : l) {
					int idx = samples.indexOf(d);
					featureBuilder.add(clusterIndex);
					featureBuilder.add(geoms.get(idx));
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}
				
				if (maxBounds == null)
					maxBounds = fc.getBounds();
				else
					maxBounds.expandToInclude(fc.getBounds());

				double[] first = l.iterator().next();
				Color color = colorMap.get(first);

				if (geoms.get(0) instanceof Polygon || geoms.get(0) instanceof MultiPolygon) {
					// mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(color, Color.BLACK, 1.0))));
					mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(color))));
				} else if (geoms.get(0) instanceof Point) {
					Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, color);
					mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPointSymbolizer(sb.createGraphic(null, mark, null)))));
				} else
					throw new RuntimeException("No layer for geometry type added");
				clusterIndex++;
			}
			
			JMapPane mp = new JMapPane();
			mp.setDoubleBuffered(true);
			mp.setMapContent(mc);
			mp.setSize(1024, 1024);
			mc.setViewport(new MapViewport(maxBounds));
														
			GTRenderer renderer = new StreamingRenderer();
			RenderingHints hints = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
			hints.put(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
			hints.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
			mp.setRenderer(renderer);
												
			Rectangle imageBounds = null;
			try {
				double heightToWidth = maxBounds.getSpan(1) / maxBounds.getSpan(0);
				imageBounds = new Rectangle(offset, offset, mp.getWidth() + offset, (int) Math.round(mp.getWidth() * heightToWidth) + offset);
				{
					/*FileOutputStream stream = new FileOutputStream("output/test.png");
					BufferedImage bufImage = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB);
					Graphics2D g = bufImage.createGraphics();
					g.drawImage(bufImage, 0, 0, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset, null);
					renderer.paint(g, imageBounds, maxBounds);

					ImageIO.write(bufImage, "PNG", stream);
					stream.flush();
					stream.close();*/
				}
				
				FileOutputStream stream = new FileOutputStream(fn);
				EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
				g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext() );
				g.setupDocument(stream, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset);
				try {
					renderer.paint(g, imageBounds, maxBounds);
				} catch( IllegalArgumentException iae ) {
					log.warn("Ignoring "+iae.getMessage());
				}
				g.finish();
				
				stream.flush();
				stream.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			//mc.dispose();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		System.out.println("int");
		for (int i = 0; i < 5; i++) {
			System.out.println(getColor(i));
		}

		System.out.println("float");
		for (float i = 0; i < 5; i++) {
			System.out.println(getColor(i));
		}
	}
}

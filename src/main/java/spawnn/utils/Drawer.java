package spawnn.utils;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.apache.xmlgraphics.java2d.ps.EPSDocumentGraphics2D;
import org.geotools.factory.CommonFactoryFinder;
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
import org.geotools.styling.Stroke;
import org.geotools.styling.Style;
import org.geotools.styling.StyleBuilder;
import org.geotools.styling.Symbolizer;
import org.geotools.swing.JMapPane;
import org.opengis.feature.type.GeometryType;
import org.opengis.filter.FilterFactory2;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.MultiPoint;
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
		
		Map<double[], Color> colorMap = ColorUtils.getColorMap(valueMap, ColorBrewer.Set3,ColorUtils.ColorClass.Equal);
		// draw
		try {
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("cluster");
			typeBuilder.add("cluster", Integer.class);
			typeBuilder.setCRS(null);

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
			// renderer.setRendererHints(hints);
			renderer.setMapContent(map);

			Rectangle imageBounds = null;
			try {
				double heightToWidth = maxBounds.getSpan(1) / maxBounds.getSpan(0);
				int imageWidth = 2000;

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

		Map<double[], Color> colorMap = ColorUtils.getColorMap(valueMap, ColorBrewer.Set3);

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
					/*
					 * FileOutputStream stream = new FileOutputStream("output/test.png"); BufferedImage bufImage = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB); Graphics2D g = bufImage.createGraphics(); g.drawImage(bufImage, 0, 0, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset, null); renderer.paint(g, imageBounds, maxBounds);
					 * 
					 * ImageIO.write(bufImage, "PNG", stream); stream.flush(); stream.close();
					 */
				}

				FileOutputStream stream = new FileOutputStream(fn);
				EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false);
				g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext());
				g.setupDocument(stream, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset);
				try {
					renderer.paint(g, imageBounds, maxBounds);
				} catch (IllegalArgumentException iae) {
					log.warn("Ignoring " + iae.getMessage());
				}
				g.finish();

				stream.flush();
				stream.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			// mc.dispose();
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

	public static void geoDrawValues(SpatialDataFrame sdf, int column, ColorBrewer cm, String fn) {
		List<Double> values = new ArrayList<Double>();
		for (double[] d : sdf.samples)
			values.add(d[column]);
		geoDrawValues(sdf.geoms, values, sdf.crs, cm, fn);
	}

	public static void geoDrawValues(List<Geometry> geoms, List<double[]> values, int fa, CoordinateReferenceSystem crs, ColorBrewer cm, String fn) {
		List<Double> l = new ArrayList<Double>();
		for (double[] d : values)
			l.add(d[fa]);
		geoDrawValues(geoms, l, crs, cm, fn);
	}

	public static void geoDrawValues(List<Geometry> geoms, List<Double> values, CoordinateReferenceSystem crs, ColorBrewer cm, String fn) {
		Map<Geometry, Double> valueMap = new HashMap<Geometry, Double>();
		for (int i = 0; i < geoms.size(); i++)
			valueMap.put(geoms.get(i), values.get(i));
		geoDraw( ColorUtils.getColorMap(valueMap, cm), crs, fn);
	}
	
	public static void geoDraw(Map<Geometry,Color> m, CoordinateReferenceSystem crs, String fn) {
		SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
		typeBuilder.setName("data");
		typeBuilder.setCRS(crs);
		typeBuilder.add("the_geom", m.keySet().iterator().next().getClass());

		SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();

			ReferencedEnvelope mapBounds = mc.getMaxBounds();
			
			// because we want always the same order of layers
			List<Color> cols = new ArrayList<Color>(new HashSet<Color>(m.values()));
			Collections.sort(cols, new Comparator<Color>() {
				@Override
				public int compare(Color o1, Color o2) {
					return Integer.compare(o1.hashCode(), o2.hashCode());
				}
			});
			
			for (Color c : cols ) { // a layer for each color
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				
				for( Entry<Geometry,Color> e : m.entrySet() ) {
					if (!c.equals(e.getValue()))
						continue;
					featureBuilder.set("the_geom", e.getKey());
					features.add(featureBuilder.buildFeature("" + features.size()));
				}

				Style style = null;
				GeometryType gt = features.getSchema().getGeometryDescriptor().getType();
				if (gt.getBinding() == Polygon.class || gt.getBinding() == MultiPolygon.class) {
					Symbolizer sym = sb.createPolygonSymbolizer(sb.createStroke(), sb.createFill(c));
					style = SLD.wrapSymbolizers(sym);
				} else if (gt.getBinding() == Point.class || gt.getBinding() == MultiPoint.class) {
					Mark mark = sb.createMark(StyleBuilder.MARK_CIRCLE, sb.createFill(c), sb.createStroke());
					Symbolizer sym = sb.createPointSymbolizer(sb.createGraphic(null, mark, null));
					style = SLD.wrapSymbolizers(sym);
				} else {
					log.warn("GeomType not supported: " + gt.getBinding());
				}
				FeatureLayer fl = new FeatureLayer(features, style);
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
			}

			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);

			Rectangle imageBounds = null;

			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 2000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));

			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);

			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void geoDrawConnections(Map<double[], Set<double[]>> mst, Map<double[],Set<double[]>> hl, int[] ga, CoordinateReferenceSystem crs, String fn) {
		FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2();
		GeometryFactory gf = new GeometryFactory();
		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();
			ReferencedEnvelope mapBounds = mc.getMaxBounds();
	
			// lines
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("lines");
				typeBuilder.setCRS(crs);
				typeBuilder.add("color",Color.class);
				typeBuilder.add("the_geom", LineString.class);
	
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				for (double[] a : mst.keySet())
					for (double[] b : mst.get(a) ) {
						LineString ls = gf.createLineString(new Coordinate[] { new Coordinate(a[ga[0]], a[ga[1]]), new Coordinate(b[ga[0]], b[ga[1]]) });
						if( mst.containsKey(b) && mst.get(b).contains(a) )
							featureBuilder.set("color",  Color.BLACK );
						else
							featureBuilder.set("color",  Color.BLUE );
						featureBuilder.set("the_geom", ls);
						features.add(featureBuilder.buildFeature("" + features.size()));
					}
				{
				Stroke stroke = sb.createStroke(ff.property("color"), ff.literal("2.0"));
				Style style = SLD.wrapSymbolizers(sb.createLineSymbolizer(stroke));	
				mc.addLayer(new FeatureLayer(features, style));
				}
				
				// highlights
				DefaultFeatureCollection hlFeatures = new DefaultFeatureCollection();
				
				if( hl != null && !hl.isEmpty() ) {
					for(double[] a : hl.keySet() ) {
							for( double[] b : hl.get(a) ) {
								LineString ls = gf.createLineString(new Coordinate[] { new Coordinate(a[ga[0]], a[ga[1]]), new Coordinate(b[ga[0]], b[ga[1]]) });
								featureBuilder.set("color",  Color.RED );
								featureBuilder.set("the_geom", ls);
								hlFeatures.add(featureBuilder.buildFeature("" + hlFeatures.size()));
						}
					}
					Stroke stroke = sb.createStroke(ff.property("color"), ff.literal("4.0"));
					Style style = SLD.wrapSymbolizers(sb.createLineSymbolizer(stroke));	
					mc.addLayer(new FeatureLayer(hlFeatures, style));
				}
			}
			
			// points
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("points");
				typeBuilder.setCRS(crs);
				typeBuilder.add("the_geom", Point.class);
	
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				Set<double[]> s = new HashSet<double[]>(mst.keySet());
				for( Set<double[]> m : mst.values() )
					s.addAll(m);
				for (double[] a :s ) {
					Point p = gf.createPoint( new Coordinate( a[ga[0]], a[ga[1]] ) );
					featureBuilder.set("the_geom", p);
					features.add(featureBuilder.buildFeature("" + features.size()));
				}
				
				Style style = SLD.wrapSymbolizers(sb.createPointSymbolizer());
				FeatureLayer fl = new FeatureLayer(features, style);
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
			}
	
			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);
		
			Rectangle imageBounds = null;
	
			mapBounds.expandBy(0.2);
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 2400;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
	
			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);
	
			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void geoDrawWeightedConnections(Map<double[], Map<double[],Double>> mst, int[] ga, CoordinateReferenceSystem crs, String fn) {
		FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2();
		GeometryFactory gf = new GeometryFactory();
		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();
			ReferencedEnvelope mapBounds = mc.getMaxBounds();
	
			// lines
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("lines");
				typeBuilder.setCRS(crs);
				typeBuilder.add("color",Color.class);
				typeBuilder.add("weight",String.class);
				typeBuilder.add("the_geom", LineString.class);
	
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				for (double[] a : mst.keySet())
					for (double[] b : mst.get(a).keySet() ) {
						LineString ls = gf.createLineString(new Coordinate[] { new Coordinate(a[ga[0]], a[ga[1]]), new Coordinate(b[ga[0]], b[ga[1]]) });
						if( mst.containsKey(b) && mst.get(b).keySet().contains(a) )
							featureBuilder.set("color",  Color.BLACK );
						else
							featureBuilder.set("color",  Color.BLUE );
						featureBuilder.set("weight", ""+(double)Math.round(1000*mst.get(a).get(b))/1000 );
						featureBuilder.set("the_geom", ls);
						features.add(featureBuilder.buildFeature("" + features.size()));
					}
				{
				Stroke stroke = sb.createStroke(ff.property("color"), ff.literal("2.0"));
				Style style = SLD.wrapSymbolizers(sb.createLineSymbolizer(stroke));
				style.featureTypeStyles().add(sb.createFeatureTypeStyle(sb.createTextSymbolizer(Color.BLACK,sb.createFont("Arial", 20),"weight")));
				mc.addLayer(new FeatureLayer(features, style));
				}
			}
			
			// points
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("points");
				typeBuilder.setCRS(crs);
				typeBuilder.add("weight",String.class);
				typeBuilder.add("the_geom", Point.class);
	
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				Set<double[]> s = new HashSet<double[]>(mst.keySet());
				for( Map<double[],Double> m : mst.values() )
					s.addAll(m.keySet());
				for (double[] a :s ) {
					Point p = gf.createPoint( new Coordinate( a[ga[0]], a[ga[1]] ) );
					featureBuilder.set("the_geom", p);
					String st = "";
					for( double d : a )
						st += (double)Math.round(1000*d)/1000+",";
					featureBuilder.set("weight", st);
					features.add(featureBuilder.buildFeature("" + features.size()));
				}
				
				Style style = SLD.wrapSymbolizers(sb.createPointSymbolizer());
				style.featureTypeStyles().add(sb.createFeatureTypeStyle(sb.createTextSymbolizer(Color.RED,sb.createFont("Arial", 20),"weight")));
				FeatureLayer fl = new FeatureLayer(features, style);
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
				mapBounds.expandBy(0.2);
			}
	
			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);
		
			Rectangle imageBounds = null;
	
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 1000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
	
			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);
	
			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void geoDrawConnections2(Map<double[], Map<double[],Double>> mst, Map<double[],Double> points, int[] ga, CoordinateReferenceSystem crs, String fn) {
		FilterFactory2 ff = CommonFactoryFinder.getFilterFactory2();
		GeometryFactory gf = new GeometryFactory();
		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();
			ReferencedEnvelope mapBounds = mc.getMaxBounds();
	
			// lines
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("lines");
				typeBuilder.setCRS(crs);
				typeBuilder.add("color",Color.class);
				typeBuilder.add("weight",Double.class);
				typeBuilder.add("the_geom", LineString.class);
	
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				for (double[] a : mst.keySet())
					for (double[] b : mst.get(a).keySet() ) {
						LineString ls = gf.createLineString(new Coordinate[] { new Coordinate(a[ga[0]], a[ga[1]]), new Coordinate(b[ga[0]], b[ga[1]]) });
						if( mst.containsKey(b) && mst.get(b).keySet().contains(a) )
							featureBuilder.set("color",  Color.BLACK );
						else
							featureBuilder.set("color",  Color.BLUE );
						featureBuilder.set("weight", mst.get(a).get(b));
						featureBuilder.set("the_geom", ls);
						features.add(featureBuilder.buildFeature("" + features.size()));
					}
				{
				Stroke stroke = sb.createStroke(ff.property("color"), ff.literal("2.0"));
				Style style = SLD.wrapSymbolizers(sb.createLineSymbolizer(stroke));
				style.featureTypeStyles().add(sb.createFeatureTypeStyle(sb.createTextSymbolizer(Color.BLACK,sb.createFont("Arial", 20),"weight")));
				mc.addLayer(new FeatureLayer(features, style));
				}
			}
			
			// points
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("points");
				typeBuilder.setCRS(crs);
				typeBuilder.add("weight",Double.class);
				typeBuilder.add("the_geom", Point.class);
	
				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				Set<double[]> s = new HashSet<double[]>(mst.keySet());
				for( Map<double[],Double> m : mst.values() )
					s.addAll(m.keySet());
				for (double[] a : s ) {
					Point p = gf.createPoint( new Coordinate( a[ga[0]], a[ga[1]] ) );
					featureBuilder.set("the_geom", p);
					featureBuilder.set("weight", points.get(a));
					features.add(featureBuilder.buildFeature("" + features.size()));
				}
				
				Style style = SLD.wrapSymbolizers(sb.createPointSymbolizer());
				style.featureTypeStyles().add(sb.createFeatureTypeStyle(sb.createTextSymbolizer(Color.RED,sb.createFont("Arial", 20),"weight")));
				
				FeatureLayer fl = new FeatureLayer(features, style);
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
			}
	
			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);
		
			Rectangle imageBounds = null;
	
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 1000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
	
			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);
	
			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void geoDrawConnectivityMap(Map<double[], Set<double[]>> cm, int[] ga, String fn) {
		// draw
		try {
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("lines");
			typeBuilder.add("the_geom", LineString.class);
	
			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
	
			StyleBuilder sb = new StyleBuilder();
			MapContent map = new MapContent();
			ReferencedEnvelope maxBounds = null;
	
			GeometryFactory gf = new GeometryFactory();
			DefaultFeatureCollection fc = new DefaultFeatureCollection();
			for (double[] a : cm.keySet()) {
				for (double[] b : cm.get(a)) {
					Geometry g = gf.createLineString(new Coordinate[] { new Coordinate(a[ga[0]], a[ga[1]]), new Coordinate(b[ga[0]], b[ga[1]]) });
					featureBuilder.add(g);
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}
			}
	
			maxBounds = fc.getBounds();
			map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createLineSymbolizer(Color.BLACK))));
	
			GTRenderer renderer = new StreamingRenderer();
			RenderingHints hints = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			hints.put(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
			hints.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
	
			renderer.setMapContent(map);
	
			Rectangle imageBounds = null;
			try {
				double heightToWidth = maxBounds.getSpan(1) / maxBounds.getSpan(0);
				int imageWidth = 1024*5;
	
				imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
				// imageBounds = new Rectangle( 0, 0, mp.getWidth(), (int)
				// Math.round(mp.getWidth() * heightToWidth));
	
				BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB);
				Graphics2D gr = image.createGraphics();
				gr.setPaint(Color.WHITE);
				gr.fill(imageBounds);
	
				renderer.paint(gr, imageBounds, maxBounds);
	
				ImageIO.write(image, "png", new File(fn));
				image.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			map.dispose();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}

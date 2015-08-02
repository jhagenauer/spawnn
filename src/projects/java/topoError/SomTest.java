package topoError;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.map.MapViewport;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.geotools.swing.JMapPane;

import regionalization.RegionUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewerUtil;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

/* Geotopo-Fehler sind ein Indikator dafür, dass die Neuronen nicht korrekt die räumlichen Verhältnisse wiederspiegeln
 * 
 */
public class SomTest {

	private static Logger log = Logger.getLogger(SomTest.class);

	public static void main(String[] args) {

		Random r = new Random();
		int T_MAX = 100000;

		File file = new File("data/redcap/Election/election2004.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(file, true);

		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();
		}

		final int[] ga = new int[] { 0, 1 };
		final int[] fa = { 7 };

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		for (int[] dims : new int[][] { {6,7} }) {

			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(dims[0], dims[1]);
			SomUtils.initRandom(grid, samples);

			for (int k = 0; k <= grid.getMaxDist(); k++ ) {
				log.debug("radius: "+k);

				BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, k );
				// BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>( fDist );
				SOM som = new SOM(new GaussKernel(grid.getMaxDist()), new LinearDecay(1.0, 0.0), grid, bmuGetter);
				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get(r.nextInt(samples.size()));
					som.train((double) t / T_MAX, x);
				}
				//log.debug("qe: " + SomUtils.getMeanQuantError(grid, bmuGetter, eDist, samples));
				//log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));

				try {
					SomUtils.printComponentPlane(grid, fa[0], ColorMode.Blues, new FileOutputStream("output/bushPCT_"+dims[0]+"x"+dims[1]+"_"+k+".png"));
					/*SomUtils.printUMatrix(grid, fDist, ColorMode.Blues, true, "output/umatrix.png");
					SomUtils.printDMatrix(grid, fDist, ColorMode.Blues, new FileOutputStream("output/dmatrix.png"));*/
					SomUtils.printGeoGrid(new int[] { 0, 1 }, grid, new FileOutputStream("output/topo_"+dims[0]+"x"+dims[1]+"_"+k+".png"));
					if( 1 != 1 )
						throw new FileNotFoundException();

					// topo error
					DescriptiveStatistics ds = new DescriptiveStatistics();
					Grid2DHex<double[]> topoErrorGrid = new Grid2DHex<double[]>(grid.getSizeOfDim(0), grid.getSizeOfDim(1));
					Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter);
					for (GridPos p1 : grid.getPositions()) {
						int c = 0;
						int nbs = 0;
						for (double[] d : bmus.get(p1)) {
							for (double[] nb : ctg.get(d)) {
								GridPos p2 = bmuGetter.getBmuPos(nb, grid);
								if (grid.dist(p1, p2) > 1) // error
									c++;
								nbs++;
							}
						}
						double v = (double) c / nbs;
						topoErrorGrid.setPrototypeAt(p1, new double[] { c, nbs, v });
						ds.addValue(v);
					}
					log.debug(ds.getMin() + "," + ds.getMean() + "," + ds.getMax() + "," + ds.getStandardDeviation());
					SomUtils.printComponentPlane(topoErrorGrid, 2, ColorMode.Blues, new FileOutputStream("output/topoError_"+dims[0]+"x"+dims[1]+"_"+k+".png"));

					// save as shape
					List<double[]> s = new ArrayList<double[]>();
					Map<double[], Double> valueMap = new HashMap<double[], Double>();
					for (double[] d : samples) {
						for (Entry<GridPos, Set<double[]>> e : bmus.entrySet()) {
							if (e.getValue().contains(d)) {
								double[] dd = topoErrorGrid.getPrototypeAt(e.getKey());
								s.add(Arrays.copyOf(dd, dd.length));
								valueMap.put(d, dd[2]);
								break;
							}
						}
					}
					//DataUtils.writeShape(s, geoms, new String[] { "c", "nbs", "ratio" }, sdf.crs, "output/topoError.shp");
					geoDrawCluster(samples, valueMap, geoms, "output/topoErrorGeo_"+dims[0]+"x"+dims[1]+"_"+k+".png", true);

				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static void geoDrawCluster(List<double[]> samples, Map<double[], Double> valueMap, List<Geometry> geoms, String fn, boolean border) {

		int offset = (int) Math.ceil((1.0 + 3) / 2);
		Map<double[], Color> colorMap = ColorBrewerUtil.valuesToColors(valueMap, ColorBrewerUtil.ColorMode.Blues);

		// draw
		try {
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("cluster");
			typeBuilder.add("color", Double.class);
			typeBuilder.add("the_geom", Polygon.class);

			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());

			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();

			for (Color c : new HashSet<Color>(colorMap.values())) {
				DefaultFeatureCollection fc = new DefaultFeatureCollection();
				for (double[] d : samples) {
					if (colorMap.get(d) != c)
						continue;

					int idx = samples.indexOf(d);
					featureBuilder.add(colorMap.get(d));
					featureBuilder.add(geoms.get(idx));
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}
				mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(c))));
			}

			// outline
			DefaultFeatureCollection fc = new DefaultFeatureCollection();
			for (double[] d : samples) {

				int idx = samples.indexOf(d);
				featureBuilder.add(colorMap.get(d));
				featureBuilder.add(geoms.get(idx));
				fc.add(featureBuilder.buildFeature("" + fc.size()));
			}

			ReferencedEnvelope maxBounds = fc.getBounds();
			mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(Color.BLACK, 0.01))));

			JMapPane mp = new JMapPane();
			mp.setDoubleBuffered(true);
			mp.setMapContent(mc);
			mp.setSize(1024, 1024);
			mc.setViewport(new MapViewport(maxBounds));

			log.debug(maxBounds);

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
					FileOutputStream stream = new FileOutputStream(fn);
					BufferedImage bufImage = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB);
					Graphics2D g = bufImage.createGraphics();
					g.drawImage(bufImage, 0, 0, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset, null);
					renderer.paint(g, imageBounds, maxBounds);

					ImageIO.write(bufImage, "PNG", stream);
					stream.flush();
					stream.close();
				}

				/*
				 * FileOutputStream stream = new FileOutputStream(fn); EPSDocumentGraphics2D g = new EPSDocumentGraphics2D(false); g.setGraphicContext(new org.apache.xmlgraphics.java2d.GraphicContext() ); g.setupDocument(stream, imageBounds.width + 2 * offset, imageBounds.height + 2 * offset); try { renderer.paint(g, imageBounds, maxBounds); } catch( IllegalArgumentException iae ) { log.warn("Ignoring "+iae.getMessage()); } g.finish();
				 * 
				 * stream.flush(); stream.close();
				 */
			} catch (IOException e) {
				e.printStackTrace();
			}
			// mc.dispose();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
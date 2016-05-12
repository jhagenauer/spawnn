package wmng.ga2;

import java.awt.Color;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
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
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.geotools.swing.JMapPane;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.gui.NGResultPanel;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class WMNGParamSensREDCAP {

	private static Logger log = Logger.getLogger(WMNGParamSensREDCAP.class);

	public static void main(String[] args) {
		
		final int T_MAX = 1500000;
		final int runs = 100;
		final int threads = 4;
		final int nrNeurons = 8;
		final Random r = new Random();
		
		final int[] ga = new int[] { 0, 1 };

		File file = new File("data/redcap/Election/election2004.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(file, true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;

		// build dist matrix and add coordinates to samples
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();

			// normed coordinates
			d[2] = p1.getX();
			d[3] = p1.getY();
			
			d[6] = d[7];
		}

		final int[] gaNormed = new int[] { 2, 3 };

		final int fa = 7; // bush pct
		final int faOrig = 6;
		final int fips = 4; // county_f basically identical to fips
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		final Dist<double[]> fDistOrig = new EuclideanDist(new int[] { faOrig }); 
		final Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.zScoreColumn(samples, fa);
		//DataUtils.zScoreGeoColumns(samples, gaNormed, gDist);
		
		List<String> ns = new ArrayList<String>( sdf.names );
		ns.set(0, "lat");
		ns.set(1, "lon");
		ns.set(2, "latNormed");
		ns.set(3, "lonNormed");
		
		ns.set(6,"bushPct");
		ns.set(7, "bushPctNormed");
		
		//DataUtils.writeCSV("output/normedData.csv", samples, ns.toArray(new String[]{}));
		
		//final Map<double[], List<double[]>> knns = GeoUtils.getKNNs(samples, gDist, rcpFieldSize);
		final Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		Map<double[], Map<double[], Double>> dCtgMap = new HashMap<double[], Map<double[], Double>>();
		for (double[] d : ctg.keySet()) {
			dCtgMap.put(d, new HashMap<double[], Double>());
			for (double[] nb : ctg.get(d))
				if( d != nb )
					dCtgMap.get(d).put(nb, 1.0);
		}
		final Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(dCtgMap);
		//DataUtils.writeDistMatrixKeyValue(dMap, samples, "output/election.wtg");

		final double[] a = WMNGParamSensREDCAP_sameCluster.getSampleByFips(samples, fips, 48383);
		final double[] b = WMNGParamSensREDCAP_sameCluster.getSampleByFips(samples, fips, 48311);

		long time = System.currentTimeMillis();
		Map<double[], List<Result>> results = new HashMap<double[], List<Result>>();
				
		List<double[]> params = new ArrayList<double[]>();
		/*double step = 0.05;
		for (double alpha = 0.0; alpha <= 1; alpha += step, alpha = Math.round(alpha * 10000) / 10000.0) {
			for (double beta = 0.0; beta <= 1; beta += step, beta = Math.round(beta * 10000) / 10000.0) {
				double[] d = new double[] { alpha, beta };
				params.add(d);
			}
		}*/
		
		// alpha so niedrig, dass context kaum eine Rolle spiel
		//params.add( new double[]{ 0.0, 0.0 } );
		
		params.add( new double[]{ 0.25, 0.25 } );
		params.add( new double[]{ 0.25, 0.75 } );
				
		params.add( new double[]{ 0.5, 0.25 } );
		params.add( new double[]{ 0.5, 0.75 } );
		
		params.add( new double[]{ 0.75, 0.25 } );
		params.add( new double[]{ 0.75, 0.75 } );
		
		/*for( int i = 0; i <= 20; i++ ) {
			params.add(new double[]{ 0.25, (double)i/20 });
			params.add(new double[]{ 0.5,  (double)i/20 });
			params.add(new double[]{ 0.75, (double)i/20 });
		}*/
				
		try {			
			for (final double[] param : params) {
				log.debug(Arrays.toString(param));

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<Result>> futures = new ArrayList<Future<Result>>();

				for (int run = 0; run < runs; run++) {
					futures.add(es.submit(new Callable<Result>() {

						@Override
						public Result call() {

							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] rs = samples.get(r.nextInt(samples.size()));
								double[] d = Arrays.copyOf(rs, rs.length * 2);
								for (int j = rs.length; j < d.length; j++)
									d[j] = r.nextDouble();
								neurons.add(d);
							}

							Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
							for (double[] d : samples)
								bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

							SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, param[0], param[1]);
							ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);
							//ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.001, bg);

							bg.bmuHistMutable = true;
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}
							bg.bmuHistMutable = false;

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
							Result r = new Result();
							r.bmus = bmus;
							//r.qe = SpaceTest.getQuantizationError(samples, bmus, fDist, rcpFieldSize, knns);
							
							int numSubs = 0;
							for( Set<double[]> s : bmus.values() )
								numSubs+=RegionUtils.getAllContiguousSubcluster(ctg, s).size();
							
							DescriptiveStatistics ds1 = new DescriptiveStatistics();
							DescriptiveStatistics ds2 = new DescriptiveStatistics();
							
							for( double[] n : ng.getNeurons() ) { 
								ds1.addValue( n[fa] );
								ds2.addValue( n[n.length/2+fa]);
							}
							
							r.m.put("QE", DataUtils.getMeanQuantizationError(bmus, fDist));
							r.m.put("QE_orig", DataUtils.getMeanQuantizationError(bmus, fDistOrig));
							r.m.put("WSS", DataUtils.getWithinSumOfSquares(bmus.values(), fDist ) );
							r.m.put("WSS_orig", DataUtils.getWithinSumOfSquares(bmus.values(), fDistOrig ) );
							r.m.put("NumRegions", (double)numSubs);
							r.m.put("ptvMean",ds1.getMean() );
							r.m.put("ptvVar", ds1.getVariance() );
							r.m.put("ctxMean", ds2.getMean() );
							r.m.put("ctxVar", ds2.getVariance() );
							r.m.put("sameCluster", WMNGParamSensREDCAP_sameCluster.sameCluster(bmus, new double[][]{a,	b,} ) ? 1.0 : 0.0 );
							
							return r;
						}
					}));
				}
				es.shutdown();
								
				results.put(param, new ArrayList<Result>());
				for (Future<Result> f : futures)
					results.get(param).add(f.get());
			}

		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		log.debug("took: " + (System.currentTimeMillis() - time) / 1000 + "s");
				
		// calc means
		Map<double[], Map<String,Double>> means = new HashMap<double[], Map<String,Double>>();
		for (Entry<double[], List<Result>> e : results.entrySet()) {
			
			Map<String,Double> mean = new HashMap<String,Double>();
			for (Result re : e.getValue())
				for( Entry<String,Double> e2 : re.m.entrySet() ) {
					if( !mean.containsKey(e2.getKey() ) )
						mean.put(e2.getKey(), 0.0 );
					mean.put(e2.getKey(),mean.get(e2.getKey())+e2.getValue() / e.getValue().size());
				}
			means.put(e.getKey(), mean);
		}

		// write means/statistics to file
		try {
			FileWriter fw = new FileWriter("output/wmng_redcap_"+runs+".csv");
			String sep = " & ";
			NumberFormat nf = NumberFormat.getNumberInstance(Locale.US);
			DecimalFormat df = (DecimalFormat)nf;
			
			List<String> keys = new ArrayList<String>(means.values().iterator().next().keySet());
			fw.write("alpha"+sep+"beta");
			for( String s : keys )
				fw.write(sep+s);
			fw.write("\n");
			
			for (Entry<double[], Map<String,Double>> e : means.entrySet()) {
				fw.write( e.getKey()[0] + sep +e.getKey()[1]);
				for ( String s : keys)
					//fw.write(sep + e.getValue().get(s) );
					fw.write(sep + df.format( e.getValue().get(s) ) );
				fw.write("\\\\ \n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.exit(1);
		
		// print files
		for( double[] param : params ) {
			for( final Result re : results.get(param) ) {
				String fn = "output/wmng_"+param[0]+"_"+param[1]+"_"+results.get(param).indexOf(re);
				fn = fn.replace(".", "");
				
				List<Set<double[]>> clust = new ArrayList<Set<double[]>>(re.bmus.values());
				Collections.sort(clust, new Comparator<Set<double[]>>() { // sort by fa
					@Override
					public int compare(Set<double[]> o1, Set<double[]> o2) {
						double[] a1 = null, a2 = null;
						for( Entry<double[],Set<double[]>> e : re.bmus.entrySet() )
							if( e.getValue() == o1 )
								a1 = e.getKey();
							else if( e.getValue() == o2 )
								a2 = e.getKey();
						return Double.compare(a1[fa], a2[fa]);
					}
				});
				//Drawer.geoDrawCluster( clust, samples, geoms, fn+".png", false);
				geoDrawClusterEPS(clust, samples, geoms, fn+".eps", false, null );
				List<double[]> l = new ArrayList<double[]>();
				l.add(a);
				l.add(b);
				geoDrawClusterEPS(clust, samples, geoms, fn+"_hili.eps", false, l );
				
				// write shape
				List<double[]> ss = new ArrayList<double[]>();
				for( double[] s : samples ) {
					int i = 0;
					for( Set<double[]> c : clust ) {
						if( c.contains(s)) 
							ss.add( new double[]{s[fa], s[faOrig], i, s[fips] } );
						i++;
					}
				}
				DataUtils.writeShape(ss, geoms, new String[]{"var","origVar","cluster","FIPS"}, sdf.crs, fn+".shp");
					
				if( params.indexOf(param) == 0 ) {
					Map<double[],Double> neuronValues = new HashMap<double[],Double>();
					for( Set<double[]> s : clust ) {
						for( Entry<double[],Set<double[]>> e : re.bmus.entrySet() )
							if( e.getValue().equals(s) ) {
								neuronValues.put(e.getKey(), (double)clust.indexOf(s));
								break;
							}
					}
					Map<double[],Color> colorMap = ColorUtils.getColorMap(neuronValues, ColorBrewer.Set3 );
					NGResultPanel.saveLegend(colorMap, neuronValues, new File("output/wmng_legend.eps"), "EPS", false, 4 );
				}
				
				break; // just one run
			}
		}
	}
	
	public static void geoDrawClusterEPS(Collection<Set<double[]>> cluster, List<double[]> samples, List<Geometry> geoms, String fn, boolean border, List<double[]>  outline ) {

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
			typeBuilder.add("the_geom", Polygon.class);
			
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
				mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(color))));
				
				clusterIndex++;
			}
			
			// outline/higlight layer
			if( outline != null && !outline.isEmpty() ) {
				DefaultFeatureCollection fc = new DefaultFeatureCollection();
				for( double[] s : outline )  {
					featureBuilder.add(clusterIndex);
					featureBuilder.add(geoms.get(samples.indexOf(s)));
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}
				mc.addLayer( new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createPolygonSymbolizer(Color.BLACK, 2.0))));
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
}

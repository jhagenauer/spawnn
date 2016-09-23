package wmng.ga2;

import java.awt.Color;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
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

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.gui.NGResultPanel;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class WMNGParamSensLISA {

	private static Logger log = Logger.getLogger(WMNGParamSensLISA.class);

	public static void main(String[] args) {
		
		final int T_MAX = 150000;
		final int runs = 256;
		final int threads = 4;
		final Random r = new Random();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("output/election_lisa.shp"), true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		
		final Map<Integer,Set<double[]>> ref = new HashMap<Integer,Set<double[]>>();
		for( double[] d : samples ) {
			int lisa_cl = (int)d[1];
			if( !ref.containsKey(lisa_cl) )
				ref.put(lisa_cl, new HashSet<double[]>() );
			ref.get(lisa_cl).add(d);
		}
			
		final int fa = 10; // bush pct
		final int fips = 7; // county_f basically identical to fips
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });

		final Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.contiguityMapToDistanceMap(GeoUtils.getContiguityMap(samples, geoms, false, false)));		
		final double[] a = WMNGParamSensREDCAP_sameCluster.getSampleByFips(samples, fips, 48383);
		final double[] b = WMNGParamSensREDCAP_sameCluster.getSampleByFips(samples, fips, 48311);

		long time = System.currentTimeMillis();
		Map<double[], List<Result>> results = new HashMap<double[], List<Result>>();
				
		List<double[]> params = new ArrayList<double[]>();
		params.add( new double[]{ 0.6, 0.0, 4 } );
		params.add( new double[]{ 0.6, 0.05, 4 } );
		params.add( new double[]{ 0.6, 0.1, 4 } );
		params.add( new double[]{ 0.65, 0.2, 12 } );
		params.add( new double[]{ 0.6, 0.05, 8 } );
		
		/*for( int nrNeurons : new int[]{ 4, 8, 12, 20 } ) {
			double step = 0.05;
			for (double alpha = 0.0; alpha <= 1; alpha += step, alpha = Math.round(alpha * 10000) / 10000.0) {
				for (double beta = 0.0; beta <= 1; beta += step, beta = Math.round(beta * 10000) / 10000.0) {
					double[] d = new double[] { alpha, beta, nrNeurons };
					params.add(d);
				}
			}
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
							for (int i = 0; i < param[2]; i++) {
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
							DecayFunction nbRate = new PowerDecay((double) neurons.size() / 2, 0.1);
							DecayFunction lrRate = new PowerDecay(0.6, 0.01);
							ContextNG ng = new ContextNG(neurons, nbRate, lrRate, bg);

							bg.bmuHistMutable = true;
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}
							bg.bmuHistMutable = false;

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
							Result r = new Result();
							r.bmus = bmus;
																					
							DescriptiveStatistics ds1 = new DescriptiveStatistics();
							DescriptiveStatistics ds2 = new DescriptiveStatistics();
							
							for( double[] n : ng.getNeurons() ) { 
								ds1.addValue( n[fa] );
								ds2.addValue( n[n.length/2+fa]);
							}
							
							r.m.put("QE", DataUtils.getMeanQuantizationError(bmus, fDist));
							r.m.put("WSS", DataUtils.getWithinSumOfSquares(bmus.values(), fDist ) );
							r.m.put("ptvMean",ds1.getMean() );
							r.m.put("ptvVar", ds1.getVariance() );
							r.m.put("ctxMean", ds2.getMean() );
							r.m.put("ctxVar", ds2.getVariance() );
							r.m.put("sameCluster", WMNGParamSensREDCAP_sameCluster.sameCluster(bmus, new double[][]{a,	b,} ) ? 1.0 : 0.0 );
							r.m.put("nmi",ClusterValidation.getNormalizedMutualInformation(ref.values(), bmus.values()));
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
			String sep = ";";
			NumberFormat nf = NumberFormat.getNumberInstance(Locale.US);
			DecimalFormat df = (DecimalFormat)nf;
			
			List<String> keys = new ArrayList<String>(means.values().iterator().next().keySet());
			fw.write("alpha"+sep+"beta"+sep+"nrNeurons");
			for( String s : keys )
				fw.write(sep+s);
			fw.write("\n");
			
			for (Entry<double[], Map<String,Double>> e : means.entrySet()) {
				fw.write( e.getKey()[0] + sep +e.getKey()[1] + sep + e.getKey()[2] );
				for ( String s : keys)
					fw.write(sep + e.getValue().get(s) );
					//fw.write(sep + df.format( e.getValue().get(s) ) );
				fw.write("\n");
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
				WMNGParamSensREDCAP.geoDrawClusterEPS(clust, samples, geoms, fn+".eps", false, null );
				List<double[]> l = new ArrayList<double[]>();
				l.add(a);
				l.add(b);
				WMNGParamSensREDCAP.geoDrawClusterEPS(clust, samples, geoms, fn+"_hili.eps", false, l );
				
				// write shape
				List<double[]> ss = new ArrayList<double[]>();
				for( double[] s : samples ) {
					int i = 0;
					for( Set<double[]> c : clust ) {
						if( c.contains(s)) 
							ss.add( new double[]{s[fa], s[fa], i, s[fips] } );
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
}

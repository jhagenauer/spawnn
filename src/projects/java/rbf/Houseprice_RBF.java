package rbf;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class Houseprice_RBF {

	private static Logger log = Logger.getLogger(Houseprice_RBF.class);

	public static void main(String[] args) {
		final Random r = new Random();

		SpatialDataFrame sd = DataUtils.readSpatialDataFrameFromShapefile(new File("data/ontario/clipped/ontario_inorg_sel_final.shp"), true);
		Map<String, DescriptiveStatistics[]> results = new HashMap<String, DescriptiveStatistics[]>();
		
		/*for( int n = 7; n <= 77; n++ ) {
			Dist<double[]> gDist = new EuclideanDist(new int[]{2,3});
			log.debug(n);
			log.debug("Moran's I (id): "+GeoUtils.getMoransI(GeoUtils.getInverseDistanceMatrix(sd.samples, gDist, 2, true), n));
			log.debug("Moran's I (10-nn): "+ GeoUtils.getMoransI( GeoUtils.knnsToWeights(GeoUtils.getKNNs(sd.samples, gDist, 10)), n));
		}*/

		//for( int n = 7; n <= 77; n++ ) {
		for( final int nrCluster : new int[]{4,9} )
		for (int n : new int[] { 33 }) {
			log.debug(nrCluster);
			// co (14), ni (33), zn (55), al (57), V (51)
			for (int nnSize : new int[] { 0, 7 }) {

				EuclideanDist dist23 = new EuclideanDist(new int[] { 2, 3 });
				List<double[]> all = new ArrayList<double[]>();
				for (double[] d : sd.samples) {

					List<double[]> nns = new ArrayList<double[]>();
					while (nns.size() < nnSize) {

						double[] nn = null;
						for (double[] d2 : sd.samples) {
							if (d == d2 || nns.contains(d2))
								continue;

							if (nn == null || dist23.dist(d, d2) < dist23.dist(nn, d))
								nn = d2;
						}
						nns.add(nn);
					}

					double[] d3 = new double[3 + nns.size()];
					d3[0] = d[2]; // x
					d3[1] = d[3]; // y
					d3[2] = d[n]; // n (target)
					for (int i = 0; i < nns.size(); i++)
						d3[i + 3] = nns.get(i)[n];
					all.add(d3);
				}
				
				//DataUtils.writeCSV("output/"+n+".csv", all, new String[]{"x","y",sd.names.get(n)});

				int[] ga = new int[] { 0, 1 };
				DataUtils.zScoreGeoColumns(all, ga, new EuclideanDist(ga)); // problem?
				//DataUtils.zScoreColumns(all, ga);

				int[] fa1 = new int[all.get(0).length - 3];
				for (int i = 0; i < fa1.length; i++)
					fa1[i] = i + 3;
				DataUtils.zScoreColumns(all, fa1);

				int fa2 = 2;
				// DataUtils.zScoreColumn(all, fa2);
				/*for( double[] d : all ) 
					d[fa2] = Math.sqrt(d[fa2]);*/
				
				final int T_MAX = 50000;
				final Dist<double[]> dist = new EuclideanDist();

				final int maxK = 10;
				int maxRuns = 16;
				//final int nrCluster = 4;

				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int j = 0; j < maxRuns; j++) {

					Collections.shuffle(all);
					final List<double[]> samples = new ArrayList<double[]>();
					final List<double[]> desired = new ArrayList<double[]>();

					for (double[] d : all) {
						double[] n2 = new double[ga.length + fa1.length];
						for (int i = 0; i < ga.length; i++)
							n2[i] = d[ga[i]];
						for (int i = 0; i < fa1.length; i++)
							n2[i + ga.length] = d[fa1[i]];

						samples.add(n2);
						desired.add(new double[] { d[2] });
					}

					for (int k = 0; k < maxK; k++) {
						final int K = k;

						futures.add(es.submit(new Callable<double[]>() {

							@Override
							public double[] call() throws Exception {
								List<double[]> training = new ArrayList<double[]>();
								List<double[]> trainingDesired = new ArrayList<double[]>();
								List<double[]> validation = new ArrayList<double[]>();
								List<double[]> validationDesired = new ArrayList<double[]>();

								for (int i = 0; i < samples.size(); i++) {
									if (K * samples.size() / maxK <= i && i < (K + 1) * samples.size() / maxK) {
										validation.add(samples.get(i));
										validationDesired.add(desired.get(i));
									} else {
										training.add(samples.get(i));
										trainingDesired.add(desired.get(i));
									}
								}

								/*Map<double[], Double> hidden = new HashMap<double[], Double>();
								Map<double[], Set<double[]>> clustering = Clustering.kMeans(samples, nrCluster, dist);
								double qe = DataUtils.getMeanQuantizationError(clustering, dist);
								for (int i = 0; i < 25; i++) {
									Map<double[], Set<double[]>> tmp = Clustering.kMeans(samples, clustering.size(), dist);
									double b = DataUtils.getMeanQuantizationError(tmp, dist);
									if (b < qe) {
										qe = b;
										clustering = tmp;
									}
								}
								
								
								for (double[] c : clustering.keySet()) {
									double d = Double.MAX_VALUE;
									for (double[] n : clustering.keySet())
										if (c != n)
											d = Math.min(d, dist.dist(c, n)*1.1 );
									hidden.put(c, d);
								}*/
								
								Sorter<double[]> s;
								s = new DefaultSorter<double[]>(dist);
								//s = new KangasSorter<double[]>(gDist, fDist, radius);
								NG ng = new NG(nrCluster, (double) nrCluster / 2, 0.01, 0.5, 0.005, samples.get(0).length, s);
								
								for( int t = 0; t < T_MAX*4; t++ ) 
									ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
								Map<double[],Set<double[]>> map = NGUtils.getBmuMapping(samples, ng.getNeurons(), s);
								
								Map<double[],Double> hidden = new HashMap<double[],Double>();
								
								// moody and darken
								/*int p = 2;
								for( double[] c : map.keySet() ) {
									Set<double[]> set = new HashSet<double[]>();
									while( set.size() < p ) {
										double[] closest = null;
										for( double[] a : samples )
											if( !set.contains(a) && ( closest == null || dist.dist(a,c) < dist.dist(closest,c) ) )
												closest = a;
										set.add(closest);
									}
									double sum = 0;
									for( double[] a : set )
										sum += Math.pow( dist.dist(a, c), 2 );
									hidden.put( c, Math.sqrt(sum)/p);
								}*/
								
								// min plus overlap, saha and keeler
								for (double[] c : map.keySet() ) {
									double d = Double.MAX_VALUE;
									for (double[] n :  map.keySet() )
										if (c != n)
											d = Math.min(d, dist.dist(c, n))*1.1;
									hidden.put(c, d);
								}
																								
								// rule of thumb, Haykins 1999
								/*double maxDist = 0;
								for( double [] a : map.keySet() )
									for( double[] b : map.keySet() )
										maxDist = Math.max( maxDist, dist.dist(a,b));
								for( double[] a : map.keySet() )
									hidden.put( a, maxDist/Math.sqrt(2*nrCluster ) );*/
								
								RBF rbf = new RBF(hidden, 1, dist, 0.05);
								for (int t = 0; t < T_MAX; t++) {
									int idx = r.nextInt(training.size());
									rbf.train(training.get(idx), trainingDesired.get(idx));
								}

								List<double[]> response = new ArrayList<double[]>();
								for (double[] x : validation)
									response.add(rbf.present(x));

								double[] r = new double[] { Meuse.getRMSE(response, validationDesired), Math.pow(Meuse.getPearson(response, validationDesired), 2), rbf.getNeurons().size() };

								return r;
							}
						}));
					}
				}

				es.shutdown();

				DescriptiveStatistics[] ds = null;

				for (Future<double[]> f : futures) {
					try {
						double[] d = f.get();

						if (ds == null) {
							ds = new DescriptiveStatistics[d.length];
							for (int i = 0; i < d.length; i++)
								ds[i] = new DescriptiveStatistics();
						}

						for (int i = 0; i < d.length; i++)
							ds[i].addValue(d[i]);
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}

				String desc = nnSize + "," + n+","+sd.names.get(n);

				StringBuffer sb = new StringBuffer();
				for (int i = 0; i < ds.length; i++)
					sb.append(ds[i].getMean() + ",");
				log.debug(desc + "," + sb.substring(0, Math.min(sb.length(), 400)));
			}
		}
	}
}

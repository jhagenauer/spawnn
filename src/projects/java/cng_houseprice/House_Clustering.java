package cng_houseprice;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.Connection;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphClustering;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiLineString;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.triangulate.DelaunayTriangulationBuilder;

public class House_Clustering {

	private static Logger log = Logger.getLogger(House_Clustering.class);

	public static enum ClusterAlgorithm {
		kmeans, kmeans_geo, ward, subcounties, tracts, skater, cng, wng, cng_wgraph, cng_graph, modul
	};

	public static void main(String[] args) {
		
		
		/*final DataFrame df = DataUtils.readDataFrameFromCSV(new File("output/house_sample.csv"), new int[] {}, true);
		final int[] ga = new int[] { 0, 1 };
		final int[] fa1 = new int[df.samples.get(0).length - 3];
		for (int i = 0; i < fa1.length; i++)
			fa1[i] = i + 2;
		final int fa2 = df.samples.get(0).length - 1;*/
				
		// lnp ~ lnarea_total + lnarea_plot + age_num + cond_house_3 + heat_3 + bath_3 + attic_dum + cellar_dum + garage_3 + terr_dum + gem_kauf_index_09 + gem_abi + gem_alter_index + ln_gem_dichte + time
		final DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] {}, true);

		final int[] ga = new int[]{0,1};
		final int[] fa1 = new int[]{72,71,75,49,53,57,58,59,63,70,17,19,23,27,74};
		final int fa2 = 8;
			
		log.debug("ga: " + Arrays.toString(ga));
		log.debug("fa1: " + Arrays.toString(fa1));
		log.debug("fa2: " + fa2);

		final Random r = new Random();
		final int T_MAX = 100000;

		final int threads = 4;
		final int maxK = 10; // cross-validation
		final int maxRun = 8;
		final int maxCluster = 20;
		final int numSubsamples = df.samples.size();
		log.debug(threads + "," + maxK + "," + maxRun + "," + maxCluster + "," + numSubsamples);

		final List<ClusterAlgorithm> methods = new ArrayList<ClusterAlgorithm>();
		//methods.add(ClusterAlgorithm.subcounties);
		methods.add(ClusterAlgorithm.kmeans);
		//methods.add(ClusterAlgorithm.kmeans_geo);
		//methods.add(ClusterAlgorithm.skater);
		methods.add(ClusterAlgorithm.cng);
		//methods.add(ClusterAlgorithm.cng_wgraph);
		//methods.add(ClusterAlgorithm.cng_graph);
		methods.add(ClusterAlgorithm.wng);
		//methods.add(ClusterAlgorithm.ward);
		//methods.add(ClusterAlgorithm.modul);

		long time = System.currentTimeMillis();
		Map<String, DescriptiveStatistics[]> results = new HashMap<String, DescriptiveStatistics[]>();

		class ExpData {
			Dist<double[]> fDist, gDist;
			List<double[]> samples, desired;
			Map<double[], Set<double[]>> cm;
			Map<Set<double[]>, Clustering.TreeNode> tree;
		}

		log.debug("Preparing exp data...");
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<ExpData>> futures = new ArrayList<Future<ExpData>>();

		for (int run = 0; run < maxRun; run++) {
			futures.add(es.submit(new Callable<ExpData>() {

				@Override
				public ExpData call() throws Exception {
					ExpData ed = new ExpData();

					List<double[]> subSamples = new ArrayList<double[]>(df.samples);
					Collections.shuffle(subSamples);
					subSamples = subSamples.subList(0, numSubsamples);

					// separate samples from desired
					ed.samples = new ArrayList<double[]>();
					ed.desired = new ArrayList<double[]>();
					for (double[] d : subSamples) {
						double[] n2 = new double[ga.length + fa1.length];
						for (int i = 0; i < ga.length; i++)
							n2[i] = d[ga[i]];					
						for (int i = 0; i < fa1.length; i++)
							n2[i + ga.length] = d[fa1[i]];
						ed.samples.add(n2);
						ed.desired.add(new double[] { d[fa2] });
					}
					int[] nfa = new int[fa1.length];
					int[] nga = new int[ga.length];	
					for (int i = 0; i < ga.length; i++)
						nga[i] = i;
					for (int i = 0; i < fa1.length; i++)
						nfa[i] = i+nga.length;
					
					ed.fDist = new EuclideanDist(nfa);
					ed.gDist = new EuclideanDist(nga);					
					
					DataUtils.zScoreColumns(ed.samples, nfa); 
					DataUtils.zScoreGeoColumns(ed.samples, ga, ed.gDist);
					
					// build cm
					ed.cm = new HashMap<double[], Set<double[]>>();
					Map<Coordinate, double[]> coords = new HashMap<Coordinate, double[]>();
					for (double[] d : ed.samples)
						coords.put(new Coordinate(d[ga[0]], d[ga[1]]), d);

					DelaunayTriangulationBuilder dtb = new DelaunayTriangulationBuilder();
					dtb.setSites(coords.keySet());
					
					MultiLineString mls = (MultiLineString) dtb.getEdges(new GeometryFactory());
					for (int i = 0; i < mls.getNumGeometries(); i++) {
						double[] a = coords.get(mls.getGeometryN(i).getCoordinates()[0]);
						double[] b = coords.get(mls.getGeometryN(i).getCoordinates()[1]);
						
						if (a == null || b == null)
							throw new RuntimeException();
						
						double d = ed.gDist.dist(a, b);
						if( d > 100000 ) {
							//log.debug(d);
							continue;
						}
						
						if (!ed.cm.containsKey(a))
							ed.cm.put(a, new HashSet<double[]>());
						if (!ed.cm.containsKey(b))
							ed.cm.put(b, new HashSet<double[]>());
						
						ed.cm.get(a).add(b);
						ed.cm.get(b).add(a);
					}
					
					/*Map<double[],Map<double[],Double>> idm = GeoUtils.getInverseDistanceMatrix(ed.samples, ed.gDist, 2, true);
					for (int i = 0; i < nfa.length; i++)
						log.debug( df.names.get(i+nga.length)+","+GeoUtils.getMoransI( idm, fa1[i] ) );*/
					//Clustering.geoDrawConnectivityMap(ed.cm, ga, "output/connections.png");
					
					// build cluster tree
					if (methods.contains(ClusterAlgorithm.ward))
						ed.tree = Clustering.getHierarchicalClusterTree(ed.cm, ed.fDist, HierarchicalClusteringType.ward);

					return ed;
				}
			}));
		}
		es.shutdown();
		try {
			es.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		}

		for (Future<ExpData> f : futures) {
			ExpData ed = null;
			try {
				ed = f.get();
			} catch (InterruptedException | ExecutionException e1) {
				e1.printStackTrace();
			}
			
			final List<double[]> samples = ed.samples, desired = ed.desired;
			final Map<double[], Set<double[]>> cm = ed.cm;
			final Map<Set<double[]>, Clustering.TreeNode> tree = ed.tree;
			final Dist<double[]> fDist = ed.fDist, gDist = ed.gDist;

			for (final ClusterAlgorithm method : methods) {
				log.debug("Method: " + method);

				class ClusterResult {
					int[] params = null;
					long time = -1;
					double wcss = 0;
					Collection<Set<double[]>> clusters = null;
				}

				log.debug("Calculating clusters...");
				ExecutorService es2 = Executors.newFixedThreadPool(threads);
				List<Future<List<ClusterResult>>> futures2 = new ArrayList<Future<List<ClusterResult>>>();

				for (int n = 2; n <= maxCluster; n++) {
					final int N = n;
					futures2.add(es2.submit(new Callable<List<ClusterResult>>() {

						@Override
						public List<ClusterResult> call() throws Exception {
							if (method == ClusterAlgorithm.kmeans || method == ClusterAlgorithm.kmeans_geo ) {

								long start = System.currentTimeMillis();
								Map<double[], Set<double[]>> bestKM = null;
								double bestSSE = 0;
								for (int i = 0; i < 1; i++) {
									Dist<double[]> dist = fDist;
									if( method == ClusterAlgorithm.kmeans_geo )
										dist = gDist;
									
									Map<double[], Set<double[]>> km = Clustering.kMeans(samples, N, dist);
									double sse = DataUtils.getSumOfSquaresError(km, dist);
									
									if (bestKM == null || sse < bestSSE) {
										bestKM = km;
										bestSSE = sse;
									}
								}

								ClusterResult cr = new ClusterResult();
								cr.params = new int[] { N, 0 };
								cr.clusters = bestKM.values();
								cr.time = System.currentTimeMillis() - start;
								cr.wcss = DataUtils.getWithinClusterSumOfSuqares(cr.clusters, fDist);

								List<ClusterResult> results = new ArrayList<ClusterResult>();
								results.add(cr);
								return results;

							} else if (method == ClusterAlgorithm.ward) {

								long start = System.currentTimeMillis();
								ClusterResult cr = new ClusterResult();
								cr.params = new int[] { N, 0 };
								cr.clusters = Clustering.cutTree(tree, N);
								cr.time = System.currentTimeMillis() - start;
								cr.wcss = DataUtils.getWithinClusterSumOfSuqares(cr.clusters, fDist);

								List<ClusterResult> results = new ArrayList<ClusterResult>();
								results.add(cr);
								return results;

							} else if (method == ClusterAlgorithm.subcounties /* || method == TRACTS */) {

								long start = System.currentTimeMillis();
								GeometryFactory gf = new GeometryFactory();
								File f;
								if (method == ClusterAlgorithm.subcounties)
									f = new File("data/census/subcounties/tl_2010_39095_cousub00.shp");
								else
									f = new File("data/census/tracts/tr39_d00.shp");
								List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(f);
								Map<Geometry, Set<double[]>> cMap = new HashMap<Geometry, Set<double[]>>();
								for (double[] d : samples) {
									Point p = gf.createPoint(new Coordinate(d[ga[0]], d[ga[1]]));

									Geometry c = null;
									for (Geometry geom : geoms) {
										if (c == null || p.distance(geom) < p.distance(c))
											c = geom;
									}

									if (!cMap.containsKey(c))
										cMap.put(c, new HashSet<double[]>());
									cMap.get(c).add(d);
								}

								ClusterResult cr = new ClusterResult();
								cr.params = new int[] { geoms.size(), 0 };
								cr.clusters = cMap.values();
								cr.time = System.currentTimeMillis() - start;
								cr.wcss = DataUtils.getWithinClusterSumOfSuqares(cr.clusters, fDist);

								List<ClusterResult> results = new ArrayList<ClusterResult>();
								results.add(cr);
								return results;

							} else if (method == ClusterAlgorithm.skater) {

								long start = System.currentTimeMillis();
								Map<double[], Set<double[]>> mst = Clustering.getMinimumSpanningTree(cm, fDist);

								ClusterResult cr = new ClusterResult();
								cr.params = new int[] { N, 0 };
								cr.clusters = Clustering.skater(mst, N - 1, fDist, 1);
								cr.time = System.currentTimeMillis() - start;
								cr.wcss = DataUtils.getWithinClusterSumOfSuqares(cr.clusters, fDist);

								List<ClusterResult> results = new ArrayList<ClusterResult>();
								results.add(cr);
								return results;

							} else if (method == ClusterAlgorithm.cng ) {

								List<ClusterResult> results = new ArrayList<ClusterResult>();
								for (int l = 1; l <= N; l++) {

									long start = System.currentTimeMillis();
									Sorter<double[]> s = new KangasSorter<double[]>(gDist, fDist, l);
									NG ng = new NG(N, (double) N / 2, 0.01, 0.5, 0.005, samples.get(0).length, s);

									for (int t = 0; t < T_MAX; t++) {
										double[] d = samples.get(r.nextInt(samples.size()));
										ng.train((double) t / T_MAX, d);
									}

									ClusterResult cr = new ClusterResult();
									cr.params = new int[] { N, l };
									cr.clusters = NGUtils.getBmuMapping(samples, ng.getNeurons(), s).values();
									cr.time = System.currentTimeMillis() - start;
									cr.wcss = DataUtils.getWithinClusterSumOfSuqares(cr.clusters, fDist);
	
									results.add(cr);
								}
								return results;
							} else if( method == ClusterAlgorithm.wng ) {
								
								List<ClusterResult> results = new ArrayList<ClusterResult>();
								for (int l = 0; l <= 100; l++ ) {

									long start = System.currentTimeMillis();
									
									Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
									map.put(fDist, 1 - (double)l/100);
									map.put(gDist, (double)l/100 );
									Sorter<double[]> s = new DefaultSorter<double[]>(new WeightedDist<double[]>(map));
									NG ng = new NG(N, (double) N / 2, 0.01, 0.5, 0.005, samples.get(0).length, s);

									for (int t = 0; t < T_MAX; t++) {
										double[] d = samples.get(r.nextInt(samples.size()));
										ng.train((double) t / T_MAX, d);
									}

									ClusterResult cr = new ClusterResult();
									cr.params = new int[] { N, l };
									cr.clusters = NGUtils.getBmuMapping(samples, ng.getNeurons(), s).values();
									cr.time = System.currentTimeMillis() - start;
									cr.wcss = DataUtils.getWithinClusterSumOfSuqares(cr.clusters, fDist);
	
									results.add(cr);
								}
								return results;
							} else if( method == ClusterAlgorithm.cng_wgraph || method == ClusterAlgorithm.cng_graph) {
								
								List<ClusterResult> results = new ArrayList<ClusterResult>();
								int neurons = 50;
								for (int l = 1; l <= 12; l++) {

									long start = System.currentTimeMillis();
									Sorter<double[]> s = new KangasSorter<double[]>(gDist, fDist, l);
									NG ng = new NG(neurons, (double) neurons / 2, 0.01, 0.5, 0.005, samples.get(0).length, s);

									for (int t = 0; t < T_MAX; t++) {
										double[] d = samples.get(r.nextInt(samples.size()));
										ng.train((double) t / T_MAX, d);
									}
									
									// CHL
									Map<Connection, Integer> conns = new HashMap<Connection, Integer>();
									for (double[] x : samples) {
										s.sort(x, ng.getNeurons());
										List<double[]> bmuList = ng.getNeurons();

										Connection c = new Connection(bmuList.get(0), bmuList.get(1));
										if (!conns.containsKey(c))
											conns.put(c, 1);
										else
											conns.put(c, conns.get(c) + 1);
									}
									
									int max = Collections.max(conns.values());
									Map<double[],Map<double[],Double>> graph = new HashMap<double[],Map<double[],Double>>();
									for( Connection c : conns.keySet() ) {
										double[] a = c.getA();
										double[] b = c.getB();
										if( !graph.containsKey(a) )
											graph.put( a, new HashMap<double[],Double>() );
										if( !graph.containsKey(b) )
											graph.put( b, new HashMap<double[],Double>() );
										if( method == ClusterAlgorithm.cng_wgraph) {
											graph.get(a).put(b, (double)conns.get(c)/max);
											graph.get(b).put(a, (double)conns.get(c)/max);
										} else if( method == ClusterAlgorithm.cng_graph ) {
											graph.get(a).put(b, 1.0);
											graph.get(b).put(a, 1.0);
										}
									}
																		
									Map<double[],Integer> map = GraphClustering.greedyOptModularity(graph, 10, N);
									Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), s);
									
									// neuron clusters  to data clusters
									List<Set<double[]>> clusters = new ArrayList<Set<double[]>>();
									for( Set<double[]> ps : GraphClustering.modulMapToCluster(map) ) {
										Set<double[]> ns = new HashSet<double[]>();
										for( double[] d : ps )
											ns.addAll(mapping.get(d));
										clusters.add(ns);
									}
									
									/*if( clusters.size() != N ) {
										log.debug("N: "+N+", l: "+l+", cs: "+clusters.size());
										clusters.clear(); // should provoke an exception when building lms, producing invalid regression (which is wanted)
									}*/
																		
									ClusterResult cr = new ClusterResult();
									cr.params = new int[] { N, l };
									cr.clusters = clusters;
									cr.time = System.currentTimeMillis() - start;
									cr.wcss = DataUtils.getWithinClusterSumOfSuqares(cr.clusters, fDist);
	
									results.add(cr);
								}
								return results;
							} else {
								return null;
							}
						}
					}));

					if (method == ClusterAlgorithm.subcounties || method == ClusterAlgorithm.tracts)
						break;
				}

				es2.shutdown();
				try {
					es2.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}

				log.debug("Calculating lms...");
				for (Future<List<ClusterResult>> f2 : futures2) {
					try {
						for (ClusterResult cr : f2.get()) {
							List<List<double[]>> cluster = new ArrayList<List<double[]>>();
							for (Set<double[]> s : cr.clusters)
								cluster.add(new ArrayList<double[]>(s));

							String desc = method + "," + cr.params[0] + "," + cr.params[1];
							
							// continue or set valid flags
							if( (int)cr.params[0] != cluster.size() ) {
								log.debug("size problem! "+(int)cr.params[0]+"!="+cluster.size()+". skipping...");
								continue;
							}

							int nc = 0;
							Map<double[], Integer> clusterMap = new HashMap<double[], Integer>(); // sample to cluster-membership
							for (List<double[]> s : cluster) {
								for (double[] d : s)
									clusterMap.put(d, nc);
								nc++;
							}

							Set<Integer> toIgnore = new HashSet<Integer>();
							toIgnore.add(0);
							toIgnore.add(1);

							double[] y = new double[samples.size()];
							double[][] x = new double[samples.size()][];
							for (int i = 0; i < samples.size(); i++) {
								double[] d = samples.get(i);

								double[] nd = getDouble(d, toIgnore); // remove geography
								x[i] = Arrays.copyOf(nd, nd.length + cluster.size() - 1);

								for (int l = 0; l < cluster.size() - 1; l++)
									x[i][nd.length + l] = clusterMap.get(d) == l ? 1 : 0; // add dummy cluster variables

								y[i] = desired.get(samples.indexOf(d))[0];
							}

							double aic = 0;
							DescriptiveStatistics rmse = new DescriptiveStatistics();
							DescriptiveStatistics isRmse = new DescriptiveStatistics(); //in-sample estimate
							DescriptiveStatistics r2 = new DescriptiveStatistics();
							boolean validCV = true, validFull = true;

							// cv
							for (int k = 0; k < maxK; k++) {
								List<double[]> trainingX = new ArrayList<double[]>();
								List<double[]> testingX = new ArrayList<double[]>();
								List<Double> trainingY = new ArrayList<Double>();
								List<Double> testingY = new ArrayList<Double>();

								// basic cv
								for (int i = 0; i < x.length; i++) {
									if (k * x.length / maxK <= i && i < (k + 1) * x.length / maxK) {
										testingX.add(x[i]);
										testingY.add(y[i]);
									} else {
										trainingX.add(x[i]);
										trainingY.add(y[i]);
									}
								}

								double[] ly = new double[trainingY.size()];
								for (int i = 0; i < trainingY.size(); i++)
									ly[i] = trainingY.get(i);

								double[][] lx = new double[trainingX.size()][];
								for (int i = 0; i < trainingX.size(); i++)
									lx[i] = trainingX.get(i);

								try { // predict for rmse and r2
									// training
									OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
									ols.setNoIntercept(false);
									ols.newSampleData(ly, lx);
									double[] beta = ols.estimateRegressionParameters();
									
									// testing
									List<double[]> response = new ArrayList<double[]>();
									List<double[]> desiredResponse = new ArrayList<double[]>();
									for (int i = 0; i < testingX.size(); i++) {
										double[] xi = testingX.get(i);

										double p = beta[0]; // intercept at beta[0]
										for (int j = 1; j < beta.length; j++)
											p += beta[j] * xi[j - 1];

										response.add(new double[] { p });
										desiredResponse.add(new double[] { testingY.get(i) });
									}
									rmse.addValue(Meuse.getRMSE(response, desiredResponse));
									r2.addValue(Math.pow(Meuse.getPearson(response, desiredResponse), 2));
								} catch (SingularMatrixException e) {
									log.debug("CV: "+desc + "," + e.getMessage());
									validCV = false;
								}
							}
							
							try { // full data for aic/ inSample-RMSE
								OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
								ols.setNoIntercept(false);
								ols.newSampleData(y, x);
								double[] beta = ols.estimateRegressionParameters();

								// aic
								double variance = ols.calculateResidualSumOfSquares() / x.length;
								// + 1 because of estimation of variance
								aic = x.length * Math.log(variance) + 2 * (ols.estimateRegressionParameters().length + 1);
								
								// in-sample rmse
								// testing
								List<double[]> response = new ArrayList<double[]>();
								List<double[]> desiredResponse = new ArrayList<double[]>();
								for (int i = 0; i < x.length; i++) {
									double[] xi = x[i];

									double p = beta[0]; // intercept at beta[0]
									for (int j = 1; j < beta.length; j++)
										p += beta[j] * xi[j - 1];

									response.add(new double[] { p });
									desiredResponse.add(new double[] { y[i] });
								}
								isRmse.addValue( Math.sqrt( Meuse.getMSE(response, desiredResponse) ) ); // not sure if this estimate really works. sources?
							} catch (SingularMatrixException e) {
								log.debug("FULL: "+desc + "," + e.getMessage());
								validFull = false;
							}

							if (!results.containsKey(desc))
								results.put(desc, new DescriptiveStatistics[] { 
										new DescriptiveStatistics(), 
										new DescriptiveStatistics(), 
										new DescriptiveStatistics(), 
										new DescriptiveStatistics(), 
										new DescriptiveStatistics(), 
										new DescriptiveStatistics(), 
										new DescriptiveStatistics(),
										new DescriptiveStatistics(), 
									});

							DescriptiveStatistics[] d = results.get(desc);
							d[0].addValue(aic);
							d[1].addValue(isRmse.getMean());
							d[2].addValue(validFull ? 1 : 0);
							d[3].addValue(rmse.getMean());
							d[4].addValue(r2.getMean());
							d[5].addValue(validCV ? 1 : 0);
							d[6].addValue(cr.wcss); // within cluster sum of squares
							d[7].addValue(cr.time / 1000.0); // seconds
						}
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}

			}

		}

		log.debug("took: " + ((System.currentTimeMillis() - time) / 1000 / 60) + " minutes.");

		try {
			String fn = "output/out_" + numSubsamples + "_" + maxRun + "_" + methods.size() + ".csv";
			FileWriter fw = new FileWriter(fn);

			fw.write("method,cluster,param,"
					+ "aic_mean,aic_stdev,"
					+ "isRmse_mean,isRmse_stdev,"
					+ "validFull,"
					+ "rmse_mean,rmse_stdev,"
					+ "r2_mean,r2_stdev,"
					+ "validCV,"
					+ "wcss_mean,wcss_stdev,"
					+ "time_mean,time_stdev\n");
			for (String d : results.keySet()) {
				DescriptiveStatistics[] ds = results.get(d);
				fw.write(d + "," + 
						ds[0].getMean() + "," + ds[0].getStandardDeviation() + "," +
						ds[1].getMean() + "," + ds[1].getStandardDeviation() + "," +
						ds[2].getMin() + "," +
						ds[3].getMean() + "," + ds[3].getStandardDeviation() + "," +
						ds[4].getMean() + "," + ds[4].getStandardDeviation() + "," +
						ds[5].getMin() + "," +
						ds[6].getMean() + "," + ds[5].getStandardDeviation() + "," +
						ds[7].getMean() + "," + ds[6].getStandardDeviation() + "\n"
				);
			}
			log.debug("written " + fn);
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static double[] getDouble(double[] d, Set<Integer> ign) {
		int numIgn = 0;
		for (int i : ign)
			if (i < d.length)
				numIgn++;

		double[] nd = new double[d.length - numIgn];
		int j = 0;
		for (int i = 0; i < d.length; i++)
			if (!ign.contains(i))
				nd[j++] = d[i];
		return nd;
	}
}

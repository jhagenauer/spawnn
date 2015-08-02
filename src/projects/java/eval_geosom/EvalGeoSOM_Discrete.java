package eval_geosom;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;

public class EvalGeoSOM_Discrete {

	private static Logger log = Logger.getLogger(EvalGeoSOM_Discrete.class);

	public static void main(String[] args) {
		final Random r = new Random();
		
		int threads = 4;
		final int MAX_RUNS = 1;
		final int T_MAX = 100000;
		final int X_DIM = 3, Y_DIM = 1;
		
		class Result {
			double param, fqe, sqe, nmi;
		}

		try {
			FileWriter fw = new FileWriter("output/overlap_"+MAX_RUNS+".csv");
			fw.write("noise,method,param,nmi,qe\n");
			
			for (double k : new double[]{ 0.1 } ) {
				
				final Map<Integer, Set<double[]>> cluster = new HashMap<Integer, Set<double[]>>();
				cluster.put(0, new HashSet<double[]>() );
				cluster.put(1, new HashSet<double[]>() );
				cluster.put(2, new HashSet<double[]>() );	
				
				final List<double[]> samples = new ArrayList<double[]>();
				
				for( int i = 0; i < 1000; i++ ) {
					double[] d = new double[]{ 0,1 };
					samples.add(d);
					cluster.get(0).add(d);
				}
				for( int i = 0; i < 200; i++ ) {
					double[] d = new double[]{1-k,0};
					samples.add(d);
					cluster.get(1).add(d);
				}
				for( int i = 0; i < 200; i++ ) {
					double[] d = new double[]{1,1};
					samples.add(d); 
					cluster.get(2).add(d);
				}
								
				//DataUtils.normalize(samples);
				DataUtils.writeCSV("output/d_"+k+".csv", samples, new String[]{"x", "y" } );
				
				int[] ga = new int[] { 0 };
				int[] fa = new int[] { 1 };

				final Dist<double[]> gDist = new EuclideanDist(ga);
				final Dist<double[]> fDist = new EuclideanDist(fa);

				for (int method : new int[] { 0, 1 }) {
					if (method == 0) { // geosom
						
						Result best = null;
						for (int radius = 0; radius <= X_DIM; radius++ ) {
							final int RADIUS = radius;
							
							ExecutorService es = Executors.newFixedThreadPool(threads);
							List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

							for (int run = 0; run < MAX_RUNS; run++) {

								futures.add(es.submit(new Callable<double[]>() {

									@Override
									public double[] call() throws Exception {				
										Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM);
										SomUtils.initRandom(grid, samples);

										KangasBmuGetter<double[]> bg = new KangasBmuGetter<double[]>(gDist, fDist, RADIUS);
										SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 0.1)), new LinearDecay(1.0, 0.005), grid, bg);
										for (int t = 0; t < T_MAX; t++) {
											double[] x = samples.get(r.nextInt(samples.size()));
											som.train((double) t / T_MAX, x);
										}
										Map<GridPos, Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bg);

										Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
										for (GridPos p : grid.getPositions()) {
											double[] pp = grid.getPrototypeAt(p);
											for (GridPos nb : grid.getNeighbours(p)) {
												if (!cm.containsKey(pp))
													cm.put(pp, new HashSet<double[]>());
												cm.get(pp).add(grid.getPrototypeAt(nb));
											}
										}
										Map<Set<double[]>, TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
										List<Set<double[]>> c = new ArrayList<Set<double[]>>();
										for (Set<double[]> s : Clustering.cutTree(tree, cluster.size())) {
											Set<double[]> set = new HashSet<double[]>();
											for (double[] p : s)
												set.addAll(mapping.get(grid.getPositionOf(p)));
											c.add(set);
										}
										

										List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
										Collections.sort(pos);
										List<double[]> l = new ArrayList<double[]>();
										for( GridPos p : pos )
											l.add(grid.getPrototypeAt(p));
										
										
										
										double nmi = ClusterValidation.getNormalizedMutualInformation(c, cluster.values());
										log.debug(RADIUS+","+nmi);
										log.debug(grid);
										
										SomUtils.printClassDist(cluster.values(), mapping, grid, "output/geosom_"+RADIUS+".png");
										DataUtils.writeCSV("output/geosom_"+RADIUS+".csv", l, new String[]{"x","y"} );
										return new double[] { SomUtils.getQuantizationError(grid, bg, fDist, samples), SomUtils.getQuantizationError(grid, bg, gDist, samples), nmi };
									}
								}));
							}
							es.shutdown();

							DescriptiveStatistics fqe = new DescriptiveStatistics();
							DescriptiveStatistics sqe = new DescriptiveStatistics();
							DescriptiveStatistics nmi = new DescriptiveStatistics();
							for (Future<double[]> f : futures) {
								double[] d = f.get();
								fqe.addValue(d[0]);
								sqe.addValue(d[1]);
								nmi.addValue(d[2]);
							}
							
							if (best == null || nmi.getMean() > best.nmi || (nmi.getMean() == best.nmi && fqe.getMean() < best.fqe)) {
								best = new Result();
								best.fqe = fqe.getMean();
								best.nmi = nmi.getMean();
								best.param = radius;
								best.sqe = sqe.getMean();
							}
						}
						log.debug(k+",geosom," + (int) best.param + "," + best.fqe + "," + best.sqe + "," + best.nmi + "");
						fw.write(k+",geosom,"+ (int) best.param + ","+best.nmi+","+best.fqe+"\n");

					} else { // wsom
						Result best = null;

						for (int w = 0; w <= 100; w++ ) {
							final double W = (double) w / 100;
							
							ExecutorService es = Executors.newFixedThreadPool(threads);
							List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

							for (int i = 0; i < MAX_RUNS; i++) {

								futures.add(es.submit(new Callable<double[]>() {

									@Override
									public double[] call() throws Exception {
										Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM,Y_DIM);
										SomUtils.initRandom(grid, samples);

										Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
										m.put(fDist, W);
										m.put(gDist, 1.0 - W);
										BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(new WeightedDist<double[]>(m));

										SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 0.1)), new LinearDecay(1.0, 0.005), grid, bg);
										for (int t = 0; t < T_MAX; t++) {
											double[] x = samples.get(r.nextInt(samples.size()));
											som.train((double) t / T_MAX, x);
										}
										Map<GridPos, Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bg);

										Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
										for (GridPos p : grid.getPositions()) {
											double[] pp = grid.getPrototypeAt(p);
											for (GridPos nb : grid.getNeighbours(p)) {
												if (!cm.containsKey(pp))
													cm.put(pp, new HashSet<double[]>());
												cm.get(pp).add(grid.getPrototypeAt(nb));
											}
										}
										Map<Set<double[]>, TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
										List<Set<double[]>> c = new ArrayList<Set<double[]>>();
										for (Set<double[]> s : Clustering.cutTree(tree, cluster.size())) {
											Set<double[]> set = new HashSet<double[]>();
											for (double[] p : s)
												set.addAll(mapping.get(grid.getPositionOf(p)));
											c.add(set);
										}
										
										List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
										Collections.sort(pos);
										List<double[]> l = new ArrayList<double[]>();
										for( GridPos p : pos )
											l.add(grid.getPrototypeAt(p));
										
										//SomUtils.printClassDist(cluster.values(), mapping, grid, "output/wsom_"+W+".png");	
										//DataUtils.writeCSV("output/wsom_"+W+".csv", l, new String[]{"x","y"} );
										
										double nmi = ClusterValidation.getNormalizedMutualInformation(c, cluster.values());
										return new double[] { SomUtils.getQuantizationError(grid, bg, fDist, samples), SomUtils.getQuantizationError(grid, bg, gDist, samples), nmi };
									}
								}));
							}
							es.shutdown();

							DescriptiveStatistics fqe = new DescriptiveStatistics();
							DescriptiveStatistics sqe = new DescriptiveStatistics();
							DescriptiveStatistics nmi = new DescriptiveStatistics();
							for (Future<double[]> f : futures) {
								double[] d = f.get();
								fqe.addValue(d[0]);
								sqe.addValue(d[1]);
								nmi.addValue(d[2]);
							}
							if (best == null || nmi.getMean() > best.nmi || (nmi.getMean() == best.nmi && fqe.getMean() < best.fqe)) {
								best = new Result();
								best.fqe = fqe.getMean();
								best.nmi = nmi.getMean();
								best.param = w;
								best.sqe = sqe.getMean();

							}
						}
						log.debug(k+",wsom," + best.param + "," + best.fqe + "," + best.sqe + "," + best.nmi + "");
						fw.write(k+",wsom," + best.param + "," + best.nmi + "," + best.fqe + "\n");
					}
				}
			}
			fw.close();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

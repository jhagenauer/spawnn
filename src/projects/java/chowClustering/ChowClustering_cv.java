package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import chowClustering.ChowClustering.PreCluster;
import chowClustering.ChowClustering.ValSet;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering_cv {

	private static Logger log = Logger.getLogger(ChowClustering_cv.class);

	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6, PRECLUST_OPT2 = 7, PRECLUST_OPT3 = 8, ASSIGN_MODE = 9;

	public static void main(String[] args) {

		int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() );
		log.debug("Threads: "+threads);

		File data = new File("data/gemeinden_gs2010/gem_dat.shp");
		//File data = new File("gem_dat.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(data, new int[]{ 1, 2 }, true);

		int[] ga = new int[] { 3, 4 };
		int[] fa =     new int[] {  7,  8,  9, 10, 19, 20 };
		int ta = 18; // bldUpRt
		int runs = 6;

		for( int i = 0; i < fa.length; i++ )
			log.debug("fa "+i+": "+sdf.names.get(fa[i]));
		log.debug("ta: "+ta+","+sdf.names.get(ta) );

		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[],Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);

		Path file = Paths.get("output/chow.csv");
		try {
			Files.createDirectories(file.getParent()); // create output dir
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		// lm 0.015897328776906357

		List<Object[]> params = new ArrayList<>();	
		/*for( int i : new int[]{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 } ) 	
			for( int j : new int[]{ 0, 2, 4, 5, 8 } )
				for( int l : new int[]{ 0,1,2,3,4,5,6,7,8,9 } ) 
					for( boolean b : new boolean[]{ true, false } )	{
						params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+2+l, PreCluster.Kmeans, i,  1, b, j });
						params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald, 1.0, gDist, fa.length+2+l, PreCluster.Kmeans, i,  1, b, j });
					}*/
		
		// to beat
		/*params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, 13, PreCluster.Kmeans, 50,  1, true, 0 });
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, 10, PreCluster.Kmeans, 100,  1, true, 0 });
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald, 1.0, gDist, 12, PreCluster.Kmeans, 800,  1, false, 0 });
		
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, 9, PreCluster.Kmeans, 80,  1, false, 0 }); // probably best
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald, 1.0, gDist, 8, PreCluster.Kmeans, 40,  1, false, 0 });
		
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, 9, PreCluster.Kmeans, 80,  1, true, 0 }); 
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald, 1.0, gDist, 15, PreCluster.Kmeans, 90,  1, true, 0 });*/
		
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, 9, PreCluster.Kmeans, 80,  1, false, 0 }); // probably best
		params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, 9, PreCluster.Kmeans, 80,  1, true, 0 }); 
		
		for( double p : new double[]{ 0.1 } ) {

		List<ValSet> vsList = new ArrayList<>();
		for( int i = 0; i < runs; i++ ) 
			vsList.add( ChowClustering.getValSet(cm, p ));
			
		// save valList
		{
			Path valFile = Paths.get("output/cvList_"+runs+".csv");
			try {
				Files.deleteIfExists(valFile);
				Files.createFile(valFile);
				
				Files.write(valFile, "idxVal,trainSample\r\n".getBytes(), StandardOpenOption.APPEND);
				for( int i = 0; i < vsList.size(); i++ ) 
					for( double[] d : vsList.get(i).samplesTrain ) {
						int j = sdf.samples.indexOf( d );						
						Files.write(valFile, (i+","+j+"\r\n").getBytes(), StandardOpenOption.APPEND);
					}					
			} catch (IOException e1) {
				e1.printStackTrace();
			}			
		}
		
		SummaryStatistics ss = new SummaryStatistics();
		for( ValSet vs : vsList ) {
			LinearModel lm = new LinearModel(vs.samplesTrain, fa, ta, false);
			
			List<Double> pred = lm.getPredictions(vs.samplesVal, fa);
			double mse = SupervisedUtils.getMSE(pred, vs.samplesVal, ta);
			ss.addValue( Math.sqrt(mse) );
		}
		log.debug("lm rmse: "+ss.getMean());
		
		for (Object[] param : params) {		
			Clustering.r.setSeed(0);
			
			String method = Arrays.toString(param)+"_"+p;
			final int idx = params.indexOf(param);
			final int assignMode = (int)param[ASSIGN_MODE];
			final double pValue = (double)param[P_VALUE];
			log.debug(idx+"/"+params.size()+","+method);
			
			for( ValSet vs : vsList ) {
			
			final int run = vsList.indexOf(vs);
			log.debug("run: "+run);
			
			List<Future<LinearModel>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);

			List<TreeNode> bestCurLayer = null;
			double bestWss = Double.POSITIVE_INFINITY;
			
			Clustering.minMode = (boolean)param[PRECLUST_OPT3];
			for( int i = 0; i < (int) param[PRECLUST_OPT2]; i++ ) {
				
				List<TreeNode> curLayer = ChowClustering.getInitCluster(vs.samplesTrain, vs.cmTrain, (PreCluster)param[PRECLUST], (int) param[PRECLUST_OPT], gDist, (int) param[MIN_OBS], threads );
				curLayer = Clustering.cutTree(curLayer, 1);
				List<Set<double[]>> cluster = Clustering.treeToCluster(curLayer);
				double wss = ClusterValidation.getWithinClusterSumOfSuqares(cluster, gDist);
				
				if( bestCurLayer == null || wss < bestWss ) {
					bestCurLayer = curLayer;
					bestWss = wss;
				}
			}
			
			Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(bestCurLayer, vs.cmTrain);
			List<TreeNode> tree = ChowClustering.getFunctionalClusterinTree(bestCurLayer, ncm, fa, ta, (HierarchicalClusteringType) param[CLUST], (ChowClustering.StructChangeTestMode) param[STRUCT_TEST], pValue, threads);
			
			int minClust = Clustering.getRoots(tree).size();
			for (int i = minClust; i <= (pValue == 1.0 ? Math.min( bestCurLayer.size(), 250) : minClust); i++ ) {
				final int nrCluster = i;
				futures.add(es.submit(new Callable<LinearModel>() {
					@Override
					public LinearModel call() throws Exception {
						List<Set<double[]>> ct = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster) );	
						LinearModel lm = new LinearModel(vs.samplesTrain, ct, fa, ta, false);
												
						// add val samples to cluster
						double sse = 0;
						for( double[] d : vs.samplesVal ) {																				
							Set<double[]> best = null;
							
							if( assignMode == 0 ) { // major vote
								int bestCount = Integer.MAX_VALUE;
								for (Set<double[]> s : ct) {
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb))
											count++;
									if (count > 0 && (best == null || count > bestCount)) {
										bestCount = count;
										best = s;
									}
								}
							} else if( assignMode == 1 ) { // closest gDist
								double[] closest = null;
								for (Set<double[]> s : ct) {
									for( double[] nb : cm.get(d) )
										if( s.contains(nb) && ( best == null || gDist.dist(nb, d) < gDist.dist(closest, d) ) ) {
											best = s;
											closest = nb;
										}
								}
							} else if( assignMode == 2 ) { // SSE-Diff
								double bestInc = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											double wssPre = DataUtils.getSumOfSquares(s, gDist);
											s.add(d);
											double inc = DataUtils.getSumOfSquares(s, gDist) - wssPre;
											s.remove(d);
											
											if (best == null || inc < bestInc) {
												best = s;
												bestInc = inc;
											}
										}
								}
							} else if( assignMode == 3 ) { // mean gDist
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									SummaryStatistics ss = new SummaryStatistics();
									for (double[] nb : cm.get(d))
										if (s.contains(nb))
											ss.addValue(gDist.dist(nb, d));
									if (ss.getN() > 0 && (best == null || ss.getMean() < bestMean)) {
										bestMean = ss.getMean();
										best = s;
									}
								}
							} else if( assignMode == 4 ) { // mean square gDist
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									SummaryStatistics ss = new SummaryStatistics();
									for (double[] nb : cm.get(d))
										if (s.contains(nb))
											ss.addValue(Math.pow(gDist.dist(nb, d), 2));
									if (ss.getN() > 0 && (best == null || ss.getMean() < bestMean)) {
										bestMean = ss.getMean();
										best = s;
									}
								}
							} else if (assignMode == 5) { // major vote + mean square gDist
								int bestCount = -1;
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									SummaryStatistics ss = new SummaryStatistics();
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											ss.addValue(Math.pow(gDist.dist(nb, d), 2));
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && ss.getMean() < bestMean)) {
										bestCount = count;
										bestMean = ss.getMean();
										best = s;
									}
								}
							} else if (assignMode == 6) { // major vote + mean gDist
								int bestCount = -1;
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									SummaryStatistics ss = new SummaryStatistics();
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											ss.addValue(gDist.dist(nb, d));
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && ss.getMean() < bestMean)) {
										bestCount = count;
										bestMean = ss.getMean();
										best = s;
									}
								}
							} else if (assignMode == 7) { // major vote + min gDist
								int bestCount = -1;
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									double dist = Double.MAX_VALUE;
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											dist = Math.min(gDist.dist(d, nb), dist);
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && dist < bestMean)) {
										bestCount = count;
										bestMean = dist;
										best = s;
									}
								}
							} else if( assignMode == 8 ) { // major vote + min SSE
								int bestCount = -1;
								double bestInc = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									double inc = Double.MAX_VALUE;
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											
											double wssPre = DataUtils.getSumOfSquares(s, gDist);
											s.add(d);
											inc = DataUtils.getSumOfSquares(s, gDist) - wssPre;
											s.remove(d);
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && inc < bestInc)) {
										bestCount = count;
										bestInc = inc;
										best = s;
									}
								}
							}
														
							List<double[]> l = new ArrayList<>();
							l.add(d);
							
							best.add(d);
							double pred = lm.getPredictions(l, fa).get(0);
							best.remove(d);
							sse += Math.pow( pred - d[ta], 2);
						}
						double rmse = Math.sqrt( sse/vs.samplesVal.size() );
												
						synchronized(this) {							
							try {
								String s = "";
								s += idx + ",\"" + method + "\"," + ct.size() + ","+rmse+"\r\n";
								Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
						
						return null;
					}
				}));
			}					

			es.shutdown();	
			try {
				es.awaitTermination(24, TimeUnit.HOURS);
			} catch (InterruptedException e1) {
				e1.printStackTrace();
			}
			System.gc();			
		}
	}
	}
	}
}

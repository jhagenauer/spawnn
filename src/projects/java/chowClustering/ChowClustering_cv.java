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

	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6, PRECLUST_OPT2 = 7, ASSIGN_MODE = 8;

	public static void main(String[] args) {

		String outputDir = "output/";
		int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() );
		log.debug("Threads: "+threads);

		//File data = new File("data/gemeinden_gs2010/gem_dat.shp");
		File data = new File("gem_dat.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(data, new int[]{ 1, 2 }, true);

		int[] ga = new int[] { 3, 4 };
		int[] fa =     new int[] {  7,  8,  9, 10, 19, 20 };
		int ta = 18; // bldUpRt
		int runs = 4;

		for( int i = 0; i < fa.length; i++ )
			log.debug("fa "+i+": "+sdf.names.get(fa[i]));
		log.debug("ta: "+ta+","+sdf.names.get(ta) );

		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[],Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);

		Path file = Paths.get(outputDir+"/chow.csv");
		try {
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		List<Object[]> params = new ArrayList<>();	
		for( int i : new int[]{ 300,400,500,600,700,800,900,1000,1100,1200 } ) {			
			for(int j = 0; j <= 4; j++ ) {
				for( int l : new int[]{ 0, 2, 4, 6, 8, 10 } ) {
					params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple,  1.0, gDist, fa.length+2+l, PreCluster.Kmeans, i,  1, j });
					params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Chow,  1.0, gDist, fa.length+2+l, PreCluster.Kmeans, i,  1, j });
					params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald,  1.0, gDist, fa.length+2+l, PreCluster.Kmeans, i,  1, j });
				}
			}
		}
								
		for( double p : new double[]{ 0.1 } ) {

		List<ValSet> vsList = new ArrayList<>();
		for( int i = 0; i < runs; i++ ) 
			vsList.add( ChowClustering.getValSet(cm, p ));
			
		// save valList
		{
			Path valFile = Paths.get(outputDir+"/cvList_"+runs+".csv");
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
			ss.addValue(mse);
		}
		log.debug("lm mse: "+ss.getMean());
		
		for( ValSet vs : vsList )
		for (Object[] param : params) {		
			Clustering.r.setSeed(0);
			
			String method = Arrays.toString(param)+"_"+p;
			final int idx = params.indexOf(param);
			final int run = vsList.indexOf(vs);
			final int assignMode = (int)param[ASSIGN_MODE];
			final double pValue = (double)param[P_VALUE];
			
			log.debug(run+","+idx+","+method);

			List<Future<LinearModel>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);

			List<TreeNode> bestCurLayer = null;
			double bestWss = Double.POSITIVE_INFINITY;
			for( int i = 0; i < (int) param[PRECLUST_OPT2]; i++ ) {
				
				List<Set<double[]>> init = ChowClustering.getInitCluster(vs.samplesTrain, vs.cmTrain, (PreCluster)param[PRECLUST], (int) param[PRECLUST_OPT], gDist );
	
				List<TreeNode> curLayer = new ArrayList<>();
				for (Set<double[]> s : init)
					curLayer.add(new TreeNode(0, 0, s));
				Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(curLayer, vs.cmTrain);
					
				// HC 1, maintain minobs
				List<TreeNode> tree1 = Clustering.getHierarchicalClusterTree(curLayer, ncm, gDist, HierarchicalClusteringType.ward, (int) param[MIN_OBS], threads );
				curLayer = Clustering.cutTree(tree1, 1);
				List<Set<double[]>> cluster = Clustering.treeToCluster(curLayer);
				double wss = ClusterValidation.getWithinClusterSumOfSuqares(cluster, gDist);
				
				if( bestCurLayer == null || wss < bestWss ) {
					bestCurLayer = curLayer;
					bestWss = wss;
					log.debug(i+","+bestWss);
				}
			}
						
			// HC 2
			log.debug("hc2");
			// update curLayer/ncm
			for( TreeNode tn : bestCurLayer )
				tn.contents = Clustering.getContents(tn);
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
						double rmse = 0;
						for( double[] d : vs.samplesVal ) {																				
							Set<double[]> best = null;
							
							if( assignMode == 0 ) {
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
							} else if( assignMode == 1 ) {
								double[] closest = null;
								for (Set<double[]> s : ct) {
									for( double[] nb : cm.get(d) )
										if( s.contains(nb) && ( best == null || gDist.dist(nb, d) < gDist.dist(closest, d) ) ) {
											best = s;
											closest = nb;
										}
								}
							} else if( assignMode == 2 ) {
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
							} else if( assignMode == 3 ) {
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
							} else if( assignMode == 4 ) {
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
							} 							
														
							List<double[]> l = new ArrayList<>();
							l.add(d);
							
							best.add(d);
							double pred = lm.getPredictions(l, fa).get(0);
							best.remove(d);
							rmse += Math.pow( pred - d[ta], 2);
						}
						rmse = Math.sqrt( rmse/vs.samplesVal.size() );
												
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

package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;

import chowClustering.ChowClustering.PreCluster;
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

public class ChowClustering_AIC {

	private static Logger log = Logger.getLogger(ChowClustering_AIC.class);

	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6, PRECLUST_OPT2 = 7, PRECLUST_OPT3 = 8;


	public static double best = Double.MAX_VALUE;		
	
	public static void main(String[] args) {

		int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() );
		log.debug("Threads: "+threads);

		File data = new File("data/gemeinden_gs2010/gem_dat.shp");
		//File data = new File("gem_dat.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(data, new int[]{ 1, 2 }, true);

		int[] ga = new int[] { 3, 4 };
		int[] fa =     new int[] {  7,  8,  9, 10, 19, 20 };
		int ta = 18; // bldUpRt

		for( int i = 0; i < fa.length; i++ )
			log.debug("fa "+i+": "+sdf.names.get(fa[i]));
		log.debug("ta: "+ta+","+sdf.names.get(ta) );

		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[],Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);

		Path file = Paths.get("output/chow_aic.csv");
		try {
			Files.createDirectories(file.getParent()); // create output dir
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		List<Object[]> params = new ArrayList<>();	
		for( int i : new int[]{ 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1400, 1500, 1600, 1700, 1800, 1900 } ) 	
			for( int l : new int[]{ 8 } ) 
				for( boolean b : new boolean[]{ true } )	{
					params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, l, PreCluster.Kmeans, i,  1, b});
					params.add(new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald, 1.0, gDist, l, PreCluster.Kmeans, i,  1, b});	
				}
		Collections.shuffle(params);
				
		{
			LinearModel lm = new LinearModel( sdf.samples, fa, ta, false);
			List<Double> pred = lm.getPredictions(sdf.samples, fa);
			double mse = SupervisedUtils.getMSE(pred, sdf.samples, ta);
			log.debug("lm aic: "+SupervisedUtils.getAICc_GWMODEL(mse, fa.length+1, sdf.samples.size()) ); // lm aic: -61856.98209268832
		}
			
		for (Object[] param : params) {		
			Clustering.r.setSeed(0);
			
			String method = Arrays.toString(param);
			final int idx = params.indexOf(param);
			final double pValue = (double)param[P_VALUE];
			log.debug( (idx+1)+"/"+params.size()+","+method);
						
			List<Future<LinearModel>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);

			List<TreeNode> bestCurLayer = null;
			double bestWss = Double.POSITIVE_INFINITY;
			
			Clustering.minMode = (boolean)param[PRECLUST_OPT3];
			for( int i = 0; i < (int) param[PRECLUST_OPT2]; i++ ) {
				
				List<TreeNode> curLayer = ChowClustering.getInitCluster(sdf.samples, cm, (PreCluster)param[PRECLUST], (int) param[PRECLUST_OPT], gDist, (int) param[MIN_OBS], threads );
				curLayer = Clustering.cutTree(curLayer, 1);
				List<Set<double[]>> cluster = Clustering.treeToCluster(curLayer);
				double wss = ClusterValidation.getWithinClusterSumOfSuqares(cluster, gDist);
				
				if( bestCurLayer == null || wss < bestWss ) {
					bestCurLayer = curLayer;
					bestWss = wss;
				}
			}
			
			Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(bestCurLayer, cm );
			List<TreeNode> tree = ChowClustering.getFunctionalClusterinTree(bestCurLayer, ncm, fa, ta, (HierarchicalClusteringType) param[CLUST], (ChowClustering.StructChangeTestMode) param[STRUCT_TEST], pValue, threads);
			
			int minClust = Clustering.getRoots(tree).size();
			for (int i = minClust; i <= (pValue == 1.0 ? Math.min( bestCurLayer.size(), 350) : minClust); i++ ) {
				final int nrCluster = i;
				futures.add(es.submit(new Callable<LinearModel>() {
					@Override
					public LinearModel call() throws Exception {
						List<Set<double[]>> ct = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster) );	
						LinearModel lm = new LinearModel( sdf.samples, ct, fa, ta, false);
						double mse = SupervisedUtils.getMSE( lm.getPredictions(sdf.samples, fa), sdf.samples, ta);
						double aic = SupervisedUtils.getAICc_GWMODEL(mse, ct.size()*(fa.length+1), sdf.samples.size());
												
						synchronized(this) {	
							
							if( aic < best ) {
								log.info("best "+aic+":"+method+","+nrCluster+","+mse+","+ClusterValidation.getWithinClusterSumOfSuqares(ct, gDist));
								best = aic;
							}
							
							try {
								String s = "";
								s += idx + ",\"" + method + "\"," + ct.size() + ","+aic+"\r\n";
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

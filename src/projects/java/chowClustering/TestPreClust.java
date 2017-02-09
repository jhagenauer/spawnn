package chowClustering;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import chowClustering.ChowClustering.PreCluster;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class TestPreClust {

	private static Logger log = Logger.getLogger(TestPreClust.class);

	public static void main(String[] args) {

		int threads = Math.max(1, Runtime.getRuntime().availableProcessors()-1);
		log.debug("Threads: " + threads);

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/gemeinden_gs2010/gem_dat.shp"), new int[] { 1, 2 }, true);

		int[] ga = new int[] { 3, 4 };

		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);

		int kmCluster = 300;
		int nrCluster = 12;
		int minObs = 10;
		
		for( boolean b : new boolean[]{ false } )
		for( PreCluster pc : new PreCluster[]{ /*PreCluster.Ward,*/ PreCluster.Kmeans, PreCluster.KmeansMinObs } ) {
			Clustering.minMode = b;
			log.debug(b+", "+pc);
			
			List<TreeNode> tree = ChowClustering.getInitCluster(sdf.samples, cm, pc, kmCluster, gDist, minObs, threads );	
			List<Set<double[]>> cluster = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster) );
			log.debug(cluster.size()+", "+ClusterValidation.getWithinClusterSumOfSuqares(cluster, gDist));
		}		
	}
}

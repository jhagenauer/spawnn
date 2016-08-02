package chowClustering;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering {
	
	private static Logger log = Logger.getLogger(ChowClustering.class);
	
	public static void main(String[] args) {
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		for( Entry<double[],Set<double[]>> e : cm.entrySet() ) // no identity
			e.getValue().remove(e.getKey());
		
		for (int i = 0; i < samples.size(); i++) {
			Coordinate c = geoms.get(i).getCentroid().getCoordinate();
			samples.get(i)[0] = c.x;
			samples.get(i)[1] = c.y;
		}
		
		Dist<double[]> gDist = new EuclideanDist(new int[] { 0,1 });
		Dist<double[]> fDist = new EuclideanDist(new int[] { 7 });
		
		 Map<double[], Set<double[]>> kmCluster = Clustering.kMeans(samples, 200, gDist);
		 int min = Integer.MAX_VALUE;
		 for( Set<double[]> s : kmCluster.values() )
			 min = Math.min(min, s.size());
		 log.debug("Min size: "+min);
		 
		 List<TreeNode> curLayer = new ArrayList<>();
		
		for( Set<double[]> s : kmCluster.values() ) {			
			TreeNode cn = new TreeNode();
			cn.age = 0;
			cn.cost = 0;
			cn.contents = s;
			curLayer.add(cn);
		}
		
		Map<TreeNode,Set<TreeNode>> ncm = new HashMap<>();
		for( TreeNode tnA : curLayer ) {
			Set<TreeNode> s = new HashSet<>();
			for( double[] a : tnA.contents )
				for( double[] nb : cm.get(a) )
					for( TreeNode tnB : curLayer )
						if( tnB.contents.contains(nb) )
							s.add(tnB);
			ncm.put(tnA, s);
		}
		
		long time = System.currentTimeMillis();
		List<TreeNode> tree = Clustering.getHierarchicalClusterTree(curLayer, ncm, fDist, HierarchicalClusteringType.ward);
		List<Set<double[]>> ct = Clustering.cutTree( tree, 7);
		log.debug("Nr cluster: "+ct.size());
		log.debug("Within sum of squares: " + DataUtils.getWithinSumOfSquares(ct, fDist)+", took: "+(System.currentTimeMillis()-time)/1000.0);		
		Drawer.geoDrawCluster(ct, samples, geoms, "output/clustering.png", true);
	}
}

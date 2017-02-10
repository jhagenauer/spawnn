package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import chowClustering.ChowClustering.PreCluster;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering_Apply {

	private static Logger log = Logger.getLogger(ChowClustering_Apply.class);

	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6,
			PRECLUST_OPT2 = 7, PRECLUST_OPT3 = 8;

	public static void main(String[] args) {

		int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
		log.debug("Threads: " + threads);

		File data = new File("data/gemeinden_gs2010/gem_dat.shp");
		// File data = new File("gem_dat.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(data, new int[] { 1, 2 }, true);

		int[] ga = new int[] { 3, 4 };
		int[] fa = new int[] { 7, 8, 9, 10, 19, 20 };
		int[] faPred = new int[] { 12, 13, 14, 15, 19, 21 };
		int ta = 18; // bldUpRt

		for (int i = 0; i < fa.length; i++)
			log.debug("fa " + i + ": " + sdf.names.get(fa[i]));
		log.debug("ta: " + ta + "," + sdf.names.get(ta));

		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);
		Map<double[], Map<double[], Double>> wcm = GeoUtils.contiguityMapToDistanceMap( cm ); 
		GeoUtils.rowNormalizeMatrix(wcm);

		Path file = Paths.get("output/chow_aic.csv");
		try {
			Files.createDirectories(file.getParent()); // create output dir
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		Object[] param = new Object[] { HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length + 2, PreCluster.Kmeans, 800, 1, true };

		Clustering.r.setSeed(0);

		String method = Arrays.toString(param);
		final double pValue = (double) param[P_VALUE];

		List<TreeNode> bestCurLayer = null;
		double bestWss = Double.POSITIVE_INFINITY;
		Clustering.minMode = (boolean) param[PRECLUST_OPT3];
		for (int i = 0; i < (int) param[PRECLUST_OPT2]; i++) {

			List<TreeNode> curLayer = ChowClustering.getInitCluster(sdf.samples, cm, (PreCluster) param[PRECLUST], (int) param[PRECLUST_OPT], gDist, (int) param[MIN_OBS], threads);
			curLayer = Clustering.cutTree(curLayer, 1);
			List<Set<double[]>> cluster = Clustering.treeToCluster(curLayer);
			double wss = ClusterValidation.getWithinClusterSumOfSuqares(cluster, gDist);

			if (bestCurLayer == null || wss < bestWss) {
				bestCurLayer = curLayer;
				bestWss = wss;
			}
		}

		Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(bestCurLayer, cm);
		List<TreeNode> tree = ChowClustering.getFunctionalClusterinTree(bestCurLayer, ncm, fa, ta, (HierarchicalClusteringType) param[CLUST], (ChowClustering.StructChangeTestMode) param[STRUCT_TEST],	pValue, threads);

		int nrCluster = 123;
		List<Set<double[]>> ct = Clustering.treeToCluster(Clustering.cutTree(tree, nrCluster));
		LinearModel lm = new LinearModel(sdf.samples, ct, fa, ta, false);
		double mse = SupervisedUtils.getMSE(lm.getPredictions(sdf.samples, fa), sdf.samples, ta);
		double aic = SupervisedUtils.getAICc_GWMODEL(mse, ct.size() * (fa.length + 1), sdf.samples.size());
		
		log.info("####### "+Arrays.toString(param)+" ########");
		log.info("#cluster: " + lm.cluster.size());
		log.info("rss: " + lm.getRSS());
		log.info("aicc: " + aic);
		log.info("mse: "+mse);
		
		Map<double[], Double> values = new HashMap<>();
		for (int i = 0; i < sdf.samples.size(); i++)
			values.put(sdf.samples.get(i), lm.getResiduals().get(i));
		log.info("moran: " + Arrays.toString( GeoUtils.getMoransIStatistics(wcm, values)));
		
		{
			SummaryStatistics ss = new SummaryStatistics();
			for( int i = 0; i < lm.cluster.size(); i++ ) 
				ss.addValue(lm.getBeta(i)[0]);
			log.info("famH span: "+(ss.getMax()-ss.getMin()));
		}
		
		{
			SummaryStatistics ss = new SummaryStatistics();
			for( int i = 0; i < lm.cluster.size(); i++ ) 
				ss.addValue(lm.getBeta(i)[lm.getBeta(i).length-1]);
			log.info("Intrcpt span: "+( ss.getMax() - ss.getMin() ) );
		}
		
		List<Double> predictions = lm.getPredictions(sdf.samples, faPred);
		List<double[]> l = new ArrayList<double[]>();
		for (double[] d : sdf.samples) {				
			double[] ns = new double[3 + fa.length + 1];

			int i = sdf.samples.indexOf(d);
			ns[0] = lm.getResiduals().get(i);
			ns[1] = predictions.get(i);
			
			for (int j = 0; j < lm.cluster.size(); j++) {
				if ( !lm.cluster.get(j).contains(d) ) 
					continue;
				
				ns[2] = j;  // cluster
					
				double[] beta = lm.getBeta(j);
				for( int k = 0; k < beta.length; k++ )
					ns[3+k] = beta[k];
				break;
			}				
			l.add(ns);
		}
		
		String[] names = new String[3 + fa.length + 1];
		names[0] = "residual";
		names[1] = "prdction";
		names[2] = "cluster";
		for( int i = 0; i < fa.length; i++ )
			names[3+i] = sdf.names.get(fa[i]);		
		names[names.length-1] = "Intrcpt";
		
		DataUtils.writeShape(l, sdf.geoms, names, sdf.crs, "output/" + method + ".shp");
		
		List<String> dissNames = new ArrayList<>();
		for (int i = 0; i < fa.length; i++)
			dissNames.add( sdf.names.get(fa[i]) );
		dissNames.add(  "Intrcpt" );
		
		dissNames.add( "numObs" );
		dissNames.add( "cluster" );
		dissNames.add( "RSS" );
		dissNames.add( "RMSD" );
		dissNames.add( "R2" );
		
		for (int i = 0; i < fa.length; i++)
			dissNames.add( "std"+sdf.names.get(fa[i]) );
		dissNames.add(  "stdIntrcpt" );
								
		List<double[]> dissSamples = new ArrayList<>();
		List<Geometry> dissGeoms = new ArrayList<>();
		for (Set<double[]> s : lm.cluster) {
			
			if( s.size() < fa.length + 1 )
				throw new RuntimeException();
							
			List<Polygon> polys = new ArrayList<>();
			for (double[] d : s) {
				int idx2 = sdf.samples.indexOf(d);
				
				MultiPolygon mp = (MultiPolygon) sdf.geoms.get(idx2);
				for( int i = 0; i < mp.getNumGeometries(); i++ )
					polys.add( (Polygon)mp.getGeometryN(i) );
			}
			
			while( true ) {
				int bestI = -1, bestJ = -1;
				for( int i = 0; i < polys.size() - 1 && bestI < 0; i++ ) {
					for( int j = i+1; j < polys.size(); j++ ) {
						
						if( !polys.get(i).intersects(polys.get(j) ) )
							continue;
						
						Geometry is = polys.get(i).intersection(polys.get(j));
						if( is instanceof Point || is instanceof MultiPoint )
							continue;
						
						bestI = i;
						bestJ = j;
						break;
					}
				}
				if( bestI < 0 )
					break;
				
				Polygon a = polys.remove(bestJ);
				Polygon b = polys.remove(bestI);
				polys.add( (Polygon)a.union(b) );
			}
			Geometry union = new GeometryFactory().createMultiPolygon(polys.toArray(new Polygon[]{}));
							
			List<double[]> li = new ArrayList<>(s);
			List<Set<double[]>> c = new ArrayList<>();
			c.add( new HashSet<>(li) );
			
			LinearModel plm = new LinearModel( li, c, fa, ta, false);
			
			List<Double> dl = new ArrayList<>();		
			for( double d : plm.getBeta(0) )
				dl.add(d);
							
			double sss = plm.getRSS();			
			dl.add( (double)s.size() );
			dl.add( (double)lm.cluster.indexOf(s) );
			dl.add( sss );
			dl.add( Math.sqrt( sss/s.size() ) ); // RMSD
			dl.add( SupervisedUtils.getR2( plm.getResiduals(), li, ta ) );
			
			LinearModel plmStd = new LinearModel( li, c, fa, ta, true);
			for( double d : plmStd.getBeta(0) )
				dl.add(d);
									
			// dl (list) to da (array)
			double[] da = new double[dl.size()];
			for( int i = 0; i < dl.size(); i++ )
				da[i] = dl.get(i);

			dissSamples.add(da);
			dissGeoms.add(union);
		}
		DataUtils.writeShape( dissSamples, dissGeoms, dissNames.toArray(new String[]{}), sdf.crs, "output/" + method + "_diss.shp" );	
		Drawer.geoDrawValues(dissGeoms,dissSamples,fa.length+3,sdf.crs,ColorBrewer.Set3,ColorClass.Equal,"output/" + method + "_cluster.png");
	}
}

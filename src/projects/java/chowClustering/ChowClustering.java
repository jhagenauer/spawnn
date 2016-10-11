package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.FDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.PrecisionModel;
import com.vividsolutions.jts.operation.union.UnaryUnionOp;
import com.vividsolutions.jts.precision.GeometryPrecisionReducer;

import landCon.LandCon;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering {

	private static Logger log = Logger.getLogger(ChowClustering.class);

	public enum StructChangeTestMode {
		Chow, AdjustedChow, Wald, ResiChow, ResiLikelihoodRatio, ResiSimple
	};

	enum Method {
		HC_CLUSTERING, TEST_CLUSTERING, HELBICH, DAY, PREDEFINED, HC
	};
	
	enum PreCluster {
		Kmeans, Ward
	}
	
	public static int METHOD = 0, CLUST = 1, STRUCT_TEST = 2, P_VALUE = 3, DIST = 4, PRECLUST = 5, MIN_OBS = 6, RUNS = 7;
	
	public static void main(String[] args) {
		int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() -1 );
		log.debug("Threads: "+threads);
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden/gem_dat.shp"), new int[]{ 1, 2 }, true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		
		/*List<double[]> samples = new ArrayList<>();
		List<Geometry> geoms = new ArrayList<>();
		for( int i = 0; i < sdf.samples.size(); i++ ) {
			if( sdf.samples.get(i)[4] > 3114857.91568308 ) {
				samples.add( sdf.samples.get(i) );
				geoms.add( sdf.geoms.get(i) );
			}
		}*/
		
		DataUtils.transform(samples, new int[] { 2 }, Transform.sqrt );
		DataUtils.transform(samples, new int[] { 6 }, Transform.log );
		DataUtils.transform(samples, new int[] { 12 }, Transform.log );
		DataUtils.transform(samples, new int[] { 13 }, Transform.log );
		
		int[] ga = new int[] { 3, 4 };
		int[] fa = new int[] { 2, 6, 7, 8, 9, 11/*, 12*/ };
		int ta = 12; // lc2000
		//int ta = 13; // lc2010
				
		for( int i = 0; i < fa.length; i++ )
			log.debug("fa "+i+": "+sdf.names.get(fa[i]));
		log.debug("ta: "+ta+","+sdf.names.get(ta) );
		
		Dist<double[]> gDist = new EuclideanDist(ga);
		Dist<double[]> fDist = new EuclideanDist(fa);

		List<double[]> samplesOrig = new ArrayList<>();
		for (double[] d : samples)
			samplesOrig.add(Arrays.copyOf(d, d.length));

		// clustering requires standardization
		DataUtils.transform(samples, fa, Transform.zScore);
		DataUtils.zScoreGeoColumns(samples, ga, gDist);
		
		Map<double[],Set<double[]>> cma = GeoUtils.getContiguityMap(samples, geoms, false, false);
						
		Path file = Paths.get("output/chow.txt");
		try {
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		List<Object[]> params = new ArrayList<>();
		
		/*params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 1.0, gDist, null, fa.length+1, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 1.0, gDist, null, fa.length+1, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+1, 1 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 1.0, gDist, null, fa.length+3, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 1.0, gDist, null, fa.length+3, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+3, 1 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 0.9, gDist, PreCluster.Kmeans, 500, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.9, gDist, PreCluster.Kmeans, 500, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 0.9, gDist, PreCluster.Kmeans, 500, 4 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 1.0, gDist, PreCluster.Kmeans, 500, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 1.0, gDist, PreCluster.Kmeans, 500, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, PreCluster.Kmeans, 500, 4 });*/
		
		//params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 0.05, gDist, PreCluster.Ward, 500, 1 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, PreCluster.Kmeans, 700, 4 }); // Benchmark
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+1, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+2, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+4, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+6, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+8, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, null, fa.length+12, 1 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 1.0, gDist, null, fa.length+1, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.95, gDist, null, fa.length+1, 1 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.9, gDist, null, fa.length+1, 1 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 1.0, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.95, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.9, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.5, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.1, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.05, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 0.01, gDist, PreCluster.Kmeans, 700, 4 });
				
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 1.0, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 0.95, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 0.9, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 0.5, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 0.1, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 0.05, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 0.01, gDist, PreCluster.Kmeans, 700, 4 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiLikelihoodRatio, 1.0, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiLikelihoodRatio, 0.95, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiLikelihoodRatio, 0.9, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiLikelihoodRatio, 0.5, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiLikelihoodRatio, 0.1, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiLikelihoodRatio, 0.05, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiLikelihoodRatio, 0.01, gDist, PreCluster.Kmeans, 700, 4 });
		
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, PreCluster.Kmeans, 700, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, PreCluster.Kmeans, 1000, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, PreCluster.Kmeans, 2000, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, PreCluster.Kmeans, 3000, 4 });
		params.add(new Object[] { Method.TEST_CLUSTERING, HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, PreCluster.Kmeans, 4000, 4 });
		
		Map<Integer, ClusterResult> re = new HashMap<>();

		log.debug("samples: "+samples.size()+", params: "+params.size());
		for (Object[] param : params) {
			Clustering.r.setSeed(0);
			int maxRuns = (int) param[RUNS];
			
			double[][] aics = new double[maxRuns][];
			double[][] rss = new double[maxRuns][];
			
			ClusterResult best = null;
			double bestAICc = Double.POSITIVE_INFINITY;

			String method = Arrays.toString(param);
			log.debug(method);

			for (int r = 0; r < maxRuns; r++) {
				log.debug("r: "+r);

				List<Set<double[]>> init = null;
				if (param[PRECLUST] != null && (PreCluster)param[PRECLUST] == PreCluster.Kmeans ) {
					List<Set<double[]>> l = new ArrayList<>(Clustering.kMeans(samples, (int) param[MIN_OBS], (Dist<double[]>) param[DIST]).values());
					init = new ArrayList<>();
					for( Set<double[]> s : l )
						if( s.isEmpty() )
							log.warn("Removing empty init cluster!");
						else
							init.add(s);					
				} else if (param[PRECLUST] != null && (PreCluster)param[PRECLUST] == PreCluster.Ward ) {
					List<TreeNode> tree = LandCon.getHierarchicalClusterTree(samples, cma, gDist, HierarchicalClusteringType.ward, threads );
					int nrCluster = samples.size();
					do 
						init = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster--) );
					while( minClusterSize(init) <= fa.length+1 );
					log.debug("init size: "+init.size()+","+minClusterSize(init));
					
				} else {
					init = new ArrayList<>();
					for (double[] d : samples) {
						Set<double[]> s = new HashSet<double[]>();
						s.add(d);
						init.add(s);
					}
				}
				
				List<Set<double[]>> cInit = new ArrayList<>();
				for( Set<double[]> s : init ) 
					if( !GeoUtils.isContiugous(cma, s) ) 
						cInit.addAll( RegionUtils.getAllContiguousSubcluster(cma, s) );
					else
						cInit.add(s);
				if( cInit.size() != init.size() ) {
					log.warn("Splitted non-contiguos clusters. Now "+cInit.size()+" instead of "+init.size()+" clusters");
					init = cInit;
				}
				
				{
				SummaryStatistics ss = new SummaryStatistics();
					for (Set<double[]> s : init)
						ss.addValue(s.size());
					log.debug("1st stats: "+ss.getMin() + "," + ss.getMean() + "," + ss.getMax());
					log.debug("1st wss: "+DataUtils.getWithinSumOfSquares(init, gDist ) );
				}
				
				List<TreeNode> curLayer = new ArrayList<>();
				for (Set<double[]> s : init) {
					TreeNode cn = new TreeNode(0, 0);
					cn.setContents(s);
					curLayer.add(cn);
				}
				
				Map<TreeNode, Set<TreeNode>> ncm = new HashMap<>();
				for (TreeNode tnA : curLayer) {
					Set<TreeNode> s = new HashSet<>();
					for (double[] a : tnA.contents)
						for (double[] nb : cma.get(a))
							for (TreeNode tnB : curLayer)
								if (tnB.contents.contains(nb))
									s.add(tnB);
					ncm.put(tnA, s);
				}
				
				// HC 1
				{
					log.debug("hc1");
					List<TreeNode> tree = Clustering.getHierarchicalClusterTree(curLayer, ncm, gDist, HierarchicalClusteringType.ward, threads );
					log.debug("tree created");
					
					int threshold;
					if (param[PRECLUST] != null )
						threshold = fa.length + 1;
					else
						threshold = (int) param[MIN_OBS];
					
					List<Set<double[]>> cluster = null;
					int nrCluster = curLayer.size();
					do {
						curLayer = Clustering.cutTree(tree, nrCluster--);
						cluster = Clustering.treeToCluster(curLayer);
					} while( minClusterSize( cluster ) < threshold );
					
					SummaryStatistics ss = new SummaryStatistics();
					for (Set<double[]> s : cluster)
						ss.addValue(s.size());
					log.debug("hc1 stats: "+ss.getMin() + "," + ss.getMean() + "," + ss.getMax());
					log.debug("hc1 wss: "+DataUtils.getWithinSumOfSquares(cluster, gDist ) );
				}
				
				// HC 2
				log.debug("hc2");
				
				// update curLayer/ncm
				for( TreeNode tn : curLayer )
					tn.contents = Clustering.getContents(tn);
				
				ncm = new HashMap<>();
				for (TreeNode tnA : curLayer) {
					Set<TreeNode> s = new HashSet<>();
					for (double[] a : tnA.contents)
						for (double[] nb : cma.get(a))
							for (TreeNode tnB : curLayer)
								if (tnB.contents.contains(nb))
									s.add(tnB);
					ncm.put(tnA, s);
				}
				
				List<TreeNode> tree = getHierarchicalClusterTree(curLayer, ncm, fa, ta, (HierarchicalClusteringType) param[CLUST], (StructChangeTestMode) param[STRUCT_TEST], (double)param[P_VALUE],threads);
				log.debug("tree created");
				
				List<Future<ClusterResult>> futures = new ArrayList<>();
				ExecutorService es = Executors.newFixedThreadPool(threads);
				for (int i = 1; i <= Math.min( curLayer.size(), 250); i += 1) {
					final int nrCluster = i;
					futures.add(es.submit(new Callable<ClusterResult>() {
						@Override
						public ClusterResult call() throws Exception {
							List<Set<double[]>> ct = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster) );
							
							List<Double> residuals = null;
							if ( minClusterSize(ct) >= fa.length + 1 ) 
								residuals = getResidualsLM(ct, samples, samples, fa, ta);									
							return new ClusterResult(ct, residuals, method);
						}
					}));
				}
				es.shutdown();
			
				aics[r] = new double[futures.size()];
				rss[r] = new double[futures.size()];
				try {
					for (Future<ClusterResult> f : futures) {
						ClusterResult cr = f.get();					
						int idx = futures.indexOf(f);		
						
						if( cr.residuals != null ) {
							double ss = getSumOfSquares(cr.residuals);
							double aicc = SupervisedUtils.getAICc_GWMODEL(ss/cr.residuals.size(), cr.cluster.size() * (fa.length+1) , samples.size());
							aics[r][idx] = aicc;
							rss[r][idx] = ss;
							
							if( best == null || aicc < bestAICc ) {
								best = cr;
								bestAICc = aicc;
							}
						} else {
							log.warn("No residuals for " +cr+", "+idx);
							aics[r][idx] = Double.NaN;
							rss[r][idx] = Double.NaN;
						}
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			int idx = params.indexOf(param);
			re.put(idx, best);

			try {
				String s = "";
				s += idx + ",aic,\"" + method + "\"," + Arrays.toString(getMean(aics)).replaceAll("\\[", "").replaceAll("\\]", "") + "\r\n";
				s += idx + ",rss,\"" + method + "\"," + Arrays.toString(getMean(rss)).replaceAll("\\[", "").replaceAll("\\]", "") + "\r\n";
				Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		log.debug("... do the rest...");
		
		Map<double[], Map<double[], Double>> wcm2 = GeoUtils.contiguityMapToDistanceMap( cma ); 
		GeoUtils.rowNormalizeMatrix(wcm2);

		// process best results of each method
		for (Entry<Integer, ClusterResult> e : re.entrySet()) {
			int idx = e.getKey();
			ClusterResult cr = e.getValue();

			Drawer.geoDrawCluster(cr.cluster, samples, geoms, "output/" + idx + ".png", false);

			List<Double> residuals = cr.residuals;
			Map<double[], Double> values = new HashMap<>();
			for (int i = 0; i < samples.size(); i++)
				values.put(samples.get(i), residuals.get(i));
			
			double ss = getSumOfSquares(residuals);
			log.info(cr.method);
			log.info("#cluster: " + cr.cluster.size());
			log.info("rss: " + ss);
			log.info("aicc: " + SupervisedUtils.getAICc_GWMODEL(ss / samples.size(), cr.cluster.size() * (fa.length + 1), samples.size()));
			log.info("r2: " + getR2(ss, samples, ta));
			log.info("moran: " + Arrays.toString( GeoUtils.getMoransIStatistics(wcm2, values)));
			
			List<double[]> l = new ArrayList<double[]>();
			for (double[] d : samples) {
				double[] ns = Arrays.copyOf(d, d.length + 1);
				for (int i = 0; i < cr.cluster.size(); i++) {
					if (cr.cluster.get(i).contains(d)) {
						ns[ns.length - 1] = i;
						break;
					}
				}
				l.add(ns);
			}
			String[] names = sdf.getNames();
			names = Arrays.copyOf(names, names.length + 1);
			names[names.length - 1] = "cluster";

			//DataUtils.writeShape(l, geoms, names, sdf.crs, "output/" + idx + ".shp");

			List<String> dissNames = new ArrayList<>();
			for (int i = 0; i < fa.length; i++)
				dissNames.add( names[fa[i]] );
			dissNames.add(  "Intrcpt" );
			
			for (int i = 0; i < fa.length; i++)
				dissNames.add( "p"+names[fa[i]] );
			dissNames.add(  "pIntrcpt" );
			
			dissNames.add(  "Cluster" );
			dissNames.add(  "SSE" );
			
			for (int i = 0; i < fa.length; i++)
				dissNames.add( "std"+names[fa[i]] );
			dissNames.add(  "stdIntrcpt" );
			
			for (int i = 0; i < fa.length; i++)
				dissNames.add( "pStd"+names[fa[i]] );
			dissNames.add(  "pStdIntrcpt" );
			
			PrecisionModel pm = new PrecisionModel(100000.0); 
						
			List<double[]> dissSamples = new ArrayList<>();
			List<Geometry> dissGeoms = new ArrayList<>();
			for (Set<double[]> s : cr.cluster) {

				List<double[]> li = new ArrayList<>();
				List<Geometry> ggs = new ArrayList<>();
				for (double[] d : s) {
					int idx2 = samples.indexOf(d);
					Geometry g = geoms.get(idx2);
					g = GeometryPrecisionReducer.reduce(g, pm);					
					ggs.add(g);
					
					double[] o = samplesOrig.get(idx2);
					li.add( Arrays.copyOf(o, o.length)); // not standardized
				}
				Geometry union = UnaryUnionOp.union(ggs);
				
				List<Double> dl = new ArrayList<>();
				if( s.size() >= fa.length + 1 ) {
					TDistribution td = new TDistribution(fa.length+2); // + intercept + error-term
					OLSMultipleLinearRegression ols = new MyOLS();
					ols.setNoIntercept(true);
					ols.newSampleData(getY(li, ta), getX(li, fa, true));
					double[] beta = ols.estimateRegressionParameters();
					double[] err = ols.estimateRegressionParametersStandardErrors();
					double[] pValue = new double[beta.length];
					for( int i = 0; i < beta.length; i++ )
						pValue[i] = 2*(1-td.cumulativeProbability( Math.abs( beta[i]/err[i] ) ) ); 
									
					for( double d : beta )
						dl.add(d);
					for( double d : pValue )
						dl.add(d);
					
					dl.add( (double)(cr.cluster.indexOf(s) ) );
					dl.add( ols.calculateResidualSumOfSquares() );
					
					DataUtils.transform(li, fa, Transform.zScore);
					DataUtils.transform(li, new int[]{ ta }, Transform.zScore);
					ols.newSampleData(getY(li, ta), getX(li, fa, true));
					double[] stdBeta = ols.estimateRegressionParameters();
					double[] stdErr = ols.estimateRegressionParametersStandardErrors();
					double[] stdPValue = new double[stdBeta.length];
					for( int i = 0; i < stdBeta.length; i++ )
						stdPValue[i] = 2*(1-td.cumulativeProbability( Math.abs( stdBeta[i]/stdErr[i] ) ) ); 
					
					for( double d : stdBeta )
						dl.add( d );
					for( double d : stdPValue )
						dl.add( d );					
				} else {
					log.warn("Could not calculate separate model!");
					for( int i = 0; i < (fa.length+1)*4+3 ; i++ )
						dl.add( Double.NaN );
				}
				
				double[] da = new double[dl.size()];
				for( int i = 0; i < dl.size(); i++ )
					da[i] = dl.get(i);

				dissSamples.add(da);
				dissGeoms.add(union);
			}
			//DataUtils.writeShape(dissSamples, dissGeoms, dissNames.toArray(new String[]{}), sdf.crs, "output/" + idx + "_diss.shp");
						
			Drawer.geoDrawValues( dissGeoms, dissSamples, 2*(fa.length+1), null, ColorBrewer.Set3, "output/" + idx + "_diss.png");
			/*for (int i = 0; i < dissNames.size(); i++) {
				String name = dissNames.get(i);
				if( name.substring(0, 3).equals("std") ) {
					
					Map<double[], Double> v = new HashMap<>();
					for (int j = 0; j < samples.size(); j++)
						v.put(dissSamples.get(i), residuals.get(i));
					GraphUtils.toWeightedGraph( GraphUtils.deriveQueenContiguitiyMap(dissSamples, dissGeoms, false), ;
					double[] m = GeoUtils.getMoransIStatistics(wcm, values);
				}
			}*/
			
			/*double ms = 0;
			for( Geometry g : dissGeoms )
				ms += g.getArea()/Math.pow(g.getLength(), 2);
			log.info("A/C^2: "+(ms/dissGeoms.size()));
			
			ms = 0;
			for( Geometry g : dissGeoms ) {
				double a = g.getArea();
				double r = g.getLength()/(Math.PI*2);
				ms += a/(Math.PI*Math.PI*r);
			}
			log.info("Isoperimetric quotient: "+(ms/dissGeoms.size()));*/
		}
		
		/*for( int i : re.keySet() )
			for( int j : re.keySet() ) {
				if( i >= j )
					continue;
				log.info("NMI "+i+","+j+": " + ClusterValidation.getNormalizedMutualInformation(re.get(i).cluster, re.get(j).cluster) );
			}*/
	}

	static List<TreeNode> getHierarchicalClusterTree(List<TreeNode> leafLayer, Map<TreeNode, Set<TreeNode>> cm, int[] fa, int ta, HierarchicalClusteringType hct, StructChangeTestMode sctm, double pValue, int threads) {

		class FlatSet<T> extends HashSet<T> {
			private static final long serialVersionUID = -1960947872875758352L;
			public int hashCode = 0;

			@Override
			public boolean add(T t) {
				hashCode += t.hashCode();
				return super.add(t);
			}

			@Override
			public boolean addAll(Collection<? extends T> c) {
				hashCode += c.hashCode();
				return super.addAll(c);
			}

			@Override
			public int hashCode() {
				return hashCode;
			}
		}

		List<TreeNode> tree = new ArrayList<>();
		Map<TreeNode, Set<double[]>> curLayer = new HashMap<>();

		int age = 0;
		for (TreeNode tn : leafLayer) {
			age = Math.max(age, tn.age);
			tree.add(tn);

			Set<double[]> content = Clustering.getContents(tn);
			curLayer.put(tn, content);
		}

		// copy of connected map
		final Map<TreeNode, Set<TreeNode>> connected = new HashMap<TreeNode, Set<TreeNode>>();
		if (cm != null)
			for (Entry<TreeNode, Set<TreeNode>> e : cm.entrySet())
				connected.put(e.getKey(), new HashSet<TreeNode>(e.getValue()));
		
		while (curLayer.size() > 1) {

			List<TreeNode> cl = new ArrayList<>(curLayer.keySet());
						
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			for (int t = 0; t < threads; t++) {
				final int T = t;

				futures.add(es.submit(new Callable<double[]>() {
					@Override
					public double[] call() throws Exception {
						int c1 = -1, c2 = -1;
						double minCost = Double.POSITIVE_INFINITY;

						for (int i = T; i < cl.size() - 1; i += threads) {
							TreeNode l1 = cl.get(i);
							
							if( !connected.containsKey(l1) )
								continue;
							Set<TreeNode> nbs = connected.get(l1);

							for (int j = i + 1; j < cl.size(); j++) {
								TreeNode l2 = cl.get(j);

								if (!nbs.contains(l2)) // disjoint
									continue;

								List<double[]> s1 = new ArrayList<>(curLayer.get(l1));
								List<double[]> s2 = new ArrayList<>(curLayer.get(l2));

								double cost = Double.NaN;
								double[] s = testStructChange(getX(s1, fa, true), getY(s1, ta), getX(s2, fa, true), getY(s2, ta), sctm);
								if( s[1] <= pValue )
									cost = s[0];
								
								if ( cost < minCost) {
									c1 = i;
									c2 = j;
									minCost = cost;
								} 
							}
						}
						return new double[] { c1, c2, minCost };
					}
				}));
			}
			es.shutdown();
			
			TreeNode c1 = null, c2 = null;
			double sMin = Double.POSITIVE_INFINITY;
			try {
				for (Future<double[]> f : futures) {
					double[] d = f.get();

					if ( d[0] >= 0 && ( c1 == null || d[2] < sMin ) ) {
						c1 = cl.get((int) d[0]);
						c2 = cl.get((int) d[1]);
						sMin = d[2];
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			
			if (c1 == null && c2 == null) { 
				log.debug("Cannot merge further: "+curLayer.size());
				return tree;
			}

			// create merge node, remove c1,c2
			Set<double[]> union = new FlatSet<double[]>();
			union.addAll(curLayer.remove(c1));
			union.addAll(curLayer.remove(c2));

			TreeNode mergeNode = new TreeNode(++age, sMin);
			mergeNode.children = Arrays.asList(new TreeNode[] { c1, c2 });

			// add nodes
			curLayer.put(mergeNode, union);
			tree.add(mergeNode);

			// update connected map
			if (connected != null) {
				// 1. merge values of c1 and c2 and put union
				Set<TreeNode> ns = connected.remove(c1);
				ns.addAll(connected.remove(c2));
				connected.put(mergeNode, ns);

				// 2. replace all values c1,c2 by merged node
				for (Set<TreeNode> s : connected.values()) {
					if (s.contains(c1) || s.contains(c2)) {
						s.remove(c1);
						s.remove(c2);
						s.add(mergeNode);
					}
				}
			}
		}
		return tree;
	}

	public static double getSumOfSquares(List<Double> residuals) {
		double s = 0;
		for (double d : residuals)
			s += Math.pow(d, 2);
		return s;
	}

	static double getR2(double ssRes, List<double[]> samples, int ta) {
		SummaryStatistics ss = new SummaryStatistics();
		for (double[] d : samples)
			ss.addValue(d[ta]);

		double mean = 0;
		for (double[] d : samples)
			mean += d[ta];
		mean /= samples.size();

		double ssTot = 0;
		for (double[] d : samples)
			ssTot += Math.pow(d[ta] - mean, 2);

		return 1.0 - ssRes / ssTot;

	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}

	public static double[][] getX(List<double[]> samples, int[] fa, boolean addIntercept) {
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] stripped = getStripped(samples.get(i), fa);
			if (addIntercept) {
				stripped = Arrays.copyOf(stripped, stripped.length + 1);
				stripped[stripped.length - 1] = 1;
			}
			x[i] = stripped;
		}
		return x;
	}
	
	public static double[][] getX(List<Set<double[]>> cluster, List<double[]> samples, int[] fa ) {
		int onlyIntercept = 0;
		for( Set<double[]> s : cluster )
			if( s.size() == 1 )
				onlyIntercept++;
		
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
						
			int curIdx = 0;
			x[i] = new double[ ( fa.length + 1 ) * ( cluster.size() - onlyIntercept ) + onlyIntercept ];
			for( Set<double[]> s : cluster ) { 				
				if( s.size() == 1 ) {
					if( s.contains(d) ) {
						x[i][curIdx] = 1.0; // only intercept
						break;
					}
					curIdx++;
				} else {
					if( s.contains(d) ) {
						double[] stripped = getStripped(d, fa);
						stripped = Arrays.copyOf(stripped, stripped.length + 1);
						stripped[stripped.length - 1] = 1;
						for (int k = 0; k < stripped.length; k++)
							x[i][curIdx + k] = stripped[k];
						break;
					}
					curIdx += fa.length + 1; 
				}
				
			}
		}
		return x;
	}

	public static double[] getY(List<double[]> samples, int ta) {
		double[] y = new double[samples.size()];
		for (int i = 0; i < samples.size(); i++)
			y[i] = samples.get(i)[ta];
		return y;
	}

	public static class MyOLS extends OLSMultipleLinearRegression {
		@Override
		protected void validateSampleData(double[][] x, double[] y) throws MathIllegalArgumentException {
			if ((x == null) || (y == null)) {
				throw new NullArgumentException();
			}
			if (x.length != y.length) {
				throw new DimensionMismatchException(y.length, x.length);
			}
			if (x.length == 0) { // Must be no y data either
				throw new NoDataException();
			}
			if ((isNoIntercept() ? x[0].length : x[0].length + 1) > x.length) {
				throw new MathIllegalArgumentException(LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS, x.length, x[0].length);
			}
		}
	}
	
	public static double[] testStructChange(double[][] x1, double[] y1, double[][] x2, double[] y2, StructChangeTestMode sctm) {
		double T1 = x1.length;
		double T2 = x2.length;
		double T = T1 + T2;
		double k = x1[0].length; // == x2[0].length

		boolean noIntercept = true;
		OLSMultipleLinearRegression ols1 = new MyOLS();
		ols1.setNoIntercept(noIntercept);
		ols1.newSampleData(y1, x1);

		OLSMultipleLinearRegression ols2 = new MyOLS();
		ols2.setNoIntercept(noIntercept);
		ols2.newSampleData(y2, x2);

		RealMatrix b1 = new Array2DRowRealMatrix(ols1.estimateRegressionParameters());
		RealMatrix b2 = new Array2DRowRealMatrix(ols2.estimateRegressionParameters());
		RealMatrix diff = b1.subtract(b2);

		RealMatrix X1 = new Array2DRowRealMatrix(x1);
		RealMatrix X2 = new Array2DRowRealMatrix(x2);

		RealMatrix m1 = MatrixUtils.inverse(X1.transpose().multiply(X1));
		RealMatrix m2 = MatrixUtils.inverse(X2.transpose().multiply(X2));
		
		if( sctm == StructChangeTestMode.ResiChow || sctm == StructChangeTestMode.ResiLikelihoodRatio || sctm == StructChangeTestMode.ResiSimple ) {
						
			double[][] xAll = new double[x1.length+x2.length][];
			for( int i = 0; i < x1.length; i++ )
				xAll[i] = x1[i];
			for( int i = 0; i < x2.length; i++ )
				xAll[x1.length+i] = x2[i];
			
			double[] yAll = new double[y1.length+y2.length];
			for( int i = 0; i < y1.length; i++ )
				yAll[i] = y1[i];
			for( int i = 0; i < y2.length; i++ )
				yAll[i+y1.length] = y2[i];
			
			OLSMultipleLinearRegression olsAll = new MyOLS();
			olsAll.setNoIntercept(noIntercept);
			olsAll.newSampleData(yAll, xAll);
			
			double s1 = ols1.calculateResidualSumOfSquares();
			double s2 = ols2.calculateResidualSumOfSquares();
			double sc = olsAll.calculateResidualSumOfSquares();
			
			if( sctm == StructChangeTestMode.ResiChow ) {
				double t = ((sc - (s1 + s2)) / k) / ((s1 + s2) / (T - 2 * k));
				FDistribution d = new FDistribution(k, T - 2 * k);				
				return new double[] { t, 1 - d.cumulativeProbability(t) }; // p-Value < 0.5 H0(equivalence) rejected, A and B not equal
			} else if(sctm == StructChangeTestMode.ResiLikelihoodRatio) {
				double base = sc; // merged
				double full = s1+s2; // separate 
				double t = (base - full) / full * (T - 2 * k);
				ChiSquaredDistribution d = new ChiSquaredDistribution(k);
				return new double[]{ t, 1-d.cumulativeProbability(t) };
				
			} else if( sctm == StructChangeTestMode.ResiSimple ) { // kind of similar to lrt
				return new double[]{ sc - (s1 + s2), 0.0};
			}
		} else if (sctm == StructChangeTestMode.Chow) {
			double s1 = 0;
			for (double d : ols1.estimateResiduals())
				s1 += d * d;

			double s2 = 0;
			for (double d : ols2.estimateResiduals())
				s2 += d * d;
			double t = diff.transpose().multiply(MatrixUtils.inverse(m1.add(m2))).multiply(diff).getEntry(0, 0) * (T - 2 * k) / (k * (s1 + s2)); // basic chow

			FDistribution d = new FDistribution(k,T - 2 * k);
			return new double[]{ t, 1 - d.cumulativeProbability(t) };
			
		} else if (sctm == StructChangeTestMode.Wald || sctm == StructChangeTestMode.AdjustedChow ) {
			double s1 = ols1.estimateErrorVariance();
			double s2 = ols2.estimateErrorVariance();
			double w = diff.transpose().multiply(MatrixUtils.inverse(m1.scalarMultiply(s1).add(m2.scalarMultiply(s2)))).multiply(diff).getEntry(0, 0);
			
			if( Double.isInfinite(w) )
				log.debug("Infinity!");
			
			if (sctm == StructChangeTestMode.Wald) {
				ChiSquaredDistribution d = new ChiSquaredDistribution(k);
				return new double[]{ w, 1-d.cumulativeProbability(w) };
			} else if (sctm == StructChangeTestMode.AdjustedChow) {
				FDistribution d = new FDistribution(k,T-2*k);
				return new double[]{w/k, 1-d.cumulativeProbability(w/k)};
			}
			
		}
		return null;
	}

	public static List<Double> getResidualsLM(double[][] xTrain, double[] yTrain, double[][] xVal, double[] yVal) {
		OLSMultipleLinearRegression ols = new MyOLS();
		ols.setNoIntercept(true);
		ols.newSampleData(yTrain, xTrain);
		double[] beta = ols.estimateRegressionParameters();

		List<Double> residuals = new ArrayList<>();
		for (int i = 0; i < xVal.length; i++) {
			double[] xi = xVal[i];

			double p = 0;
			for (int j = 0; j < beta.length; j++)
				p += beta[j] * xi[j];

			residuals.add(yVal[i] - p);
		}

		return residuals;
	}

	public static List<Double> getResidualsLM(List<Set<double[]>> cluster, List<double[]> samplesTrain, List<double[]> samplesVal, int[] fa, int ta) {
		List<double[]> s = new ArrayList<double[]>();
		List<Double> residuals = new ArrayList<Double>();

		for (Set<double[]> c : cluster) {
			List<double[]> isTrain = new ArrayList<>(samplesTrain);
			isTrain.retainAll(c);

			List<double[]> isVal = new ArrayList<>(samplesVal);
			isVal.retainAll(c);
			s.addAll(isVal);

			residuals.addAll(getResidualsLM(getX(isTrain, fa, true), getY(isTrain, ta), getX(isVal, fa, true), getY(isVal, ta)));
		}

		// sort residuals
		List<Double> sortedResiduals = new ArrayList<Double>();
		for (double[] d : samplesVal) {
			int idx = s.indexOf(d);
			sortedResiduals.add(residuals.get(idx));
		}
		return sortedResiduals;
	}

	static class ClusterResult {
		ClusterResult(List<Set<double[]>> cluster, List<Double> residuals, String method) {
			this.cluster = cluster;
			this.method = method;
			this.residuals = residuals;
		}

		String method;
		List<Set<double[]>> cluster = null;
		List<Double> residuals = null;

		@Override
		public String toString() {
			return method + "," + cluster.size();
		}
	}
	
	public static int minClusterSize( Collection<Set<double[]>> ct ) {
		int min = Integer.MAX_VALUE;
		for( Set<double[]> s : ct )
			min = Math.min(s.size(), min);
		return min;
	}

	public static double[] getMean(double[][] e) {
		double[] mean = new double[e[0].length];
		for (int r = 0; r < e.length; r++)
			for (int i = 0; i < e[r].length; i++)
				mean[i] += e[r][i] / e.length;
		return mean;
	}
}

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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.FDistribution;
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
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering {

	private static Logger log = Logger.getLogger(ChowClustering.class);

	public enum StructChangeTestMode {
		Chow, AdjustedChow, Wald, ResiChow, ResiLikelihoodRatio, ResiSimple, ResiAICc_GWModel, ResiAICc, ResiAIC_GWModel, ResiAIC, CV, F_TEST
	};
	
	enum PreCluster {
		Kmeans, Ward
	}
	
	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6, RUNS = 7;
	
	static PiecewiseLM best = null;
	static Double bestAICc = Double.POSITIVE_INFINITY;
			
	public static void main(String[] args) {
				
		int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() -1 );
		log.debug("Threads: "+threads);
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden_gs2010/gem_dat.shp"), new int[]{ 1, 2 }, true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		
		int[] ga = new int[] { 3, 4 };
		int[] fa = new int[] { 7,  8,  9, 10, 19, 20 };
		int[] faPred = new int[] { 12, 13, 14, 15, 19, 21 };
		int ta = 18; // bldUpRt
				
		for( int i = 0; i < fa.length; i++ )
			log.debug("fa "+i+": "+sdf.names.get(fa[i]));
		log.debug("ta: "+ta+","+sdf.names.get(ta) );
				
		Dist<double[]> gDist = new EuclideanDist(ga);
		
		Map<double[],Set<double[]>> cma = GeoUtils.getContiguityMap(samples, geoms, false, false);
		Map<double[], Map<double[], Double>> wcm2 = GeoUtils.contiguityMapToDistanceMap( cma ); 
		GeoUtils.rowNormalizeMatrix(wcm2);
						
		Path file = Paths.get("output/chow.txt");
		try {
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
						
		List<Object[]> params = new ArrayList<>();	
		int runs = 16;
		
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.CV, 10.0, gDist, fa.length + 1 + (int)Math.ceil( (double)(fa.length+1)/10 ), PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.CV, 5.0, gDist, fa.length + 1 + (int)Math.ceil( (double)(fa.length+1)/5 ) , PreCluster.Kmeans, 900, runs });
		
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+1, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+2, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+3, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+4, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+5, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+6, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiSimple, 1.0, gDist, fa.length+7, PreCluster.Kmeans, 900, runs });
				
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.Chow, 1.0, gDist, fa.length+1, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.Wald, 1.0, gDist, fa.length+2, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiAICc, 1.0, gDist, fa.length+2, PreCluster.Kmeans, 900, runs });
		params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.ResiAICc_GWModel, 1.0, gDist, fa.length+2, PreCluster.Kmeans, 900, runs }); // +2 works somehow
								
		Map<Integer, PiecewiseLM> re = new HashMap<>();

		log.debug("samples: "+samples.size()+", params: "+params.size());
		for (Object[] param : params) {
			int idx = params.indexOf(param);
			
			Clustering.r.setSeed(0);
			int maxRuns = (int) param[RUNS];

			String method = Arrays.toString(param);
			log.debug(method);
			
			best = null;
			bestAICc = Double.POSITIVE_INFINITY;

			List<Future<PiecewiseLM>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);
			
			for (int r = 0; r < maxRuns; r++) {
				log.debug("run "+r);
				
				List<Set<double[]>> init = null;
				if (param[PRECLUST] != null && (PreCluster)param[PRECLUST] == PreCluster.Kmeans ) {
					List<Set<double[]>> l = new ArrayList<>(Clustering.kMeans(samples, (int) param[PRECLUST_OPT], (Dist<double[]>) param[DIST]).values());
					init = new ArrayList<>();
					for( Set<double[]> s : l )
						if( s.isEmpty() )
							log.warn("Removing empty init cluster!");
						else
							init.add(s);					
				} else if (param[PRECLUST] != null && (PreCluster)param[PRECLUST] == PreCluster.Ward ) {
					List<TreeNode> tree = Clustering.getHierarchicalClusterTree( cma, gDist, HierarchicalClusteringType.ward  );
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
					log.warn(cInit.size()+" contiguous instead of "+init.size()+" clusters");
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
					long time = System.currentTimeMillis();
					log.debug("hc1");
					List<TreeNode> tree = Clustering.getHierarchicalClusterTree(curLayer, ncm, gDist, HierarchicalClusteringType.ward, (int) param[MIN_OBS], threads );
					log.debug("took: "+(System.currentTimeMillis()-time)/1000.0);
					
					curLayer = Clustering.cutTree(tree, 1);
					List<Set<double[]>> cluster = Clustering.treeToCluster(curLayer);
									
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
				
				long time = System.currentTimeMillis();
				List<TreeNode> tree = getHierarchicalClusterTree(curLayer, ncm, fa, ta, (HierarchicalClusteringType) param[CLUST], (StructChangeTestMode) param[STRUCT_TEST], (double)param[P_VALUE],threads);
				log.debug("took: "+(System.currentTimeMillis()-time)/1000.0);
								
				for (int i = 1; i <= Math.min( curLayer.size(), 250); i += 1) {
					final int nrCluster = i;
					futures.add(es.submit(new Callable<PiecewiseLM>() {
						@Override
						public PiecewiseLM call() throws Exception {
							List<Set<double[]>> ct = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster) );	
							PiecewiseLM cr = new PiecewiseLM(samples, ct, method, fa, ta );
							
							double aicc = cr.getAICc();
														
							Map<double[], Double> values = new HashMap<>();
							for (int i = 0; i < samples.size(); i++)
								values.put(samples.get(i), cr.getResiduals().get(i));
							double moran = GeoUtils.getMoransI(wcm2, values);
																	
							double famh = 0;
							{
								SummaryStatistics ss = new SummaryStatistics();
								for( double[] beta : cr.betas ) 
									ss.addValue(beta[0]);
								famh = ss.getMax()-ss.getMin();
							}
							
							double intrcpt = 0;
							{
								SummaryStatistics ss = new SummaryStatistics();
								for( double[] beta : cr.betas ) 
									ss.addValue(beta[beta.length-1]);
								intrcpt = ss.getMax()-ss.getMin();
							}
							
							synchronized(this) {
								if( best == null || aicc < bestAICc ) {
									best = cr;
									bestAICc = aicc;
								}
								
								try {
									String s = "";
									s += idx + ",\"" + method + "\"," + cr.cluster.size() +","+ aicc + ","+moran+","+famh+","+intrcpt+"\r\n";
									Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
							return null;
						}
					}));
				}					
			}
			es.shutdown();	
			try {
				es.awaitTermination(24, TimeUnit.HOURS);
			} catch (InterruptedException e1) {
				e1.printStackTrace();
			}
			re.put(idx, best);
			System.gc();
		}
		
		// process best results of each method
		PiecewiseLM best = null;
		for (Entry<Integer, PiecewiseLM> e : re.entrySet()) {
			int idx = e.getKey();
			PiecewiseLM cr = e.getValue();
						
			double aicc = cr.getAICc();
			log.info(cr.method);
			log.info("#cluster: " + cr.cluster.size());
			log.info("rss: " + cr.getRSS());
			log.info("aicc: " + aicc);
			log.info("r2: " + getR2(cr.getRSS(), samples, ta));
			
			if( best == null || aicc < best.getAICc() )
				best = cr;
						
			Map<double[], Double> values = new HashMap<>();
			for (int i = 0; i < samples.size(); i++)
				values.put(samples.get(i), cr.getResiduals().get(i));
			log.info("moran: " + Arrays.toString( GeoUtils.getMoransIStatistics(wcm2, values)));
			
			{
				SummaryStatistics ss = new SummaryStatistics();
				for( double[] beta : cr.betas ) 
					ss.addValue(beta[0]);
				log.debug("famH span: "+(ss.getMax()-ss.getMin()));
			}
			
			{
				SummaryStatistics ss = new SummaryStatistics();
				for( double[] beta : cr.betas ) 
					ss.addValue(beta[beta.length-1]);
				log.debug("Intrcpt span: "+(ss.getMax()-ss.getMin()));
			}
			
			List<Double> predictions = cr.getPredictions(samples, faPred);
			List<double[]> l = new ArrayList<double[]>();
			for (double[] d : samples) {
				double[] ns = Arrays.copyOf(d, d.length + 3);

				int i = samples.indexOf(d);
				ns[ns.length - 3] = cr.getResiduals().get(i);
				ns[ns.length - 2] = predictions.get(i);
				
				for (int j = 0; i < cr.cluster.size(); j++) { // cluster
					if (cr.cluster.get(j).contains(d)) {
						ns[ns.length - 1] = j; 
						break;
					}
				}
				
				l.add(ns);
			}
			
			String[] names = sdf.getNames();
			names = Arrays.copyOf( names, names.length + 3 );
			names[names.length - 3] = "residual";
			names[names.length - 2] = "prdction";
			names[names.length - 1] = "cluster";
			
			DataUtils.writeShape(l, geoms, names, sdf.crs, "output/" + idx + ".shp");
			
			List<String> dissNames = new ArrayList<>();
			for (int i = 0; i < fa.length; i++)
				dissNames.add( names[fa[i]] );
			dissNames.add(  "Intrcpt" );
			
			dissNames.add(  "numObs" );
			dissNames.add(  "cluster" );
			dissNames.add(  "RSS" );
			dissNames.add(  "RMSD" );
			dissNames.add( "R2" );
			
			for (int i = 0; i < fa.length; i++)
				dissNames.add( "std"+names[fa[i]] );
			dissNames.add(  "stdIntrcpt" );
									
			List<double[]> dissSamples = new ArrayList<>();
			List<Geometry> dissGeoms = new ArrayList<>();
			for (Set<double[]> s : cr.cluster) {
				
				if( s.size() < fa.length + 1 )
					throw new RuntimeException();
								
				List<Polygon> polys = new ArrayList<>();
				for (double[] d : s) {
					int idx2 = samples.indexOf(d);
					
					MultiPolygon mp = (MultiPolygon) geoms.get(idx2);
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
				
				PiecewiseLM plm = new PiecewiseLM( li, c, cr.method, fa, ta);
				
				List<Double> dl = new ArrayList<>();		
				for( double d : plm.betas.get(0) )
					dl.add(d);
								
				double sss = plm.getRSS();			
				dl.add( (double)s.size() );
				dl.add( (double)cr.cluster.indexOf(s) );
				dl.add( sss );
				dl.add( Math.sqrt( sss/s.size() ) ); // RMSD
				
				double mean = 0;
				for( double[] d : samples )
					mean += d[ta]/samples.size();
				double ssTot = 0;
				for (double[] d : samples )
					ssTot += Math.pow( d[ta] - mean, 2);
				dl.add( 1.0 - sss / ssTot ); // R2
				
				List<double[]> nli = new ArrayList<>();
				for( double[] d : li )
					nli.add( Arrays.copyOf(d, d.length));
				DataUtils.transform(nli, fa, Transform.zScore);
								
				List<Set<double[]>> nc = new ArrayList<>();
				nc.add( new HashSet<>(nli) );
				
				PiecewiseLM plmStd = new PiecewiseLM( nli, nc, cr.method, fa, ta);
				for( double d : plmStd.betas.get(0) )
					dl.add(d);
										
				// dl (list) to da (array)
				double[] da = new double[dl.size()];
				for( int i = 0; i < dl.size(); i++ )
					da[i] = dl.get(i);

				dissSamples.add(da);
				dissGeoms.add(union);
			}
			DataUtils.writeShape( dissSamples, dissGeoms, dissNames.toArray(new String[]{}), sdf.crs, "output/" + idx + "_diss.shp" );	
		}
		log.info("best: "+best.method+", "+best.getAICc());
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
								
								int folds = (int)pValue;
								int repeats = 1;
								double cost = Double.NaN;
								if( sctm == StructChangeTestMode.CV ) {
									SummaryStatistics ss1 = new SummaryStatistics();									
									for ( final Entry<List<Integer>, List<Integer>> cvEntry : SupervisedUtils.getCVList(folds, repeats, s1.size() ) ) {
										List<double[]> samplesTrain = new ArrayList<double[]>();
										for (int k : cvEntry.getKey()) 
											samplesTrain.add(s1.get(k));
										
										List<double[]> samplesVal = new ArrayList<double[]>();
										for (int k : cvEntry.getValue()) 
											samplesVal.add(s1.get(k));
										
										List<Set<double[]>> l = new ArrayList<>();
										l.add( new HashSet<>(samplesTrain));
										PiecewiseLM plm = new PiecewiseLM(samplesTrain, l, "", fa, ta);
										
										List<Double> pred = plm.getPredictions(samplesVal, fa);
										List<Double> err = new ArrayList<>();
										for( int k = 0; k < samplesVal.size(); k++ )
											err.add( pred.get(k) - samplesVal.get(k)[ta] );
										ss1.addValue( ChowClustering.getSumOfSquares(err) );
									}
									
									SummaryStatistics ss2 = new SummaryStatistics();									
									for ( final Entry<List<Integer>, List<Integer>> cvEntry : SupervisedUtils.getCVList(folds, repeats, s2.size() ) ) {
										List<double[]> samplesTrain = new ArrayList<double[]>();
										for (int k : cvEntry.getKey()) 
											samplesTrain.add(s2.get(k));
										
										List<double[]> samplesVal = new ArrayList<double[]>();
										for (int k : cvEntry.getValue()) 
											samplesVal.add(s2.get(k));
										
										List<Set<double[]>> l = new ArrayList<>();
										l.add( new HashSet<>(samplesTrain));
										PiecewiseLM plm = new PiecewiseLM(samplesTrain, l, "", fa, ta);
										
										List<Double> pred = plm.getPredictions(samplesVal, fa);
										List<Double> err = new ArrayList<>();
										for( int k = 0; k < samplesVal.size(); k++ )
											err.add( pred.get(k) - samplesVal.get(k)[ta] );
										ss2.addValue( ChowClustering.getSumOfSquares(err) );						
									}
									
									SummaryStatistics ssm = new SummaryStatistics();	
									List<double[]> merged = new ArrayList<>(s1);
									merged.addAll(s2);
									for ( final Entry<List<Integer>, List<Integer>> cvEntry : SupervisedUtils.getCVList(folds, repeats, merged.size() ) ) {
										List<double[]> samplesTrain = new ArrayList<double[]>();
										for (int k : cvEntry.getKey()) 
											samplesTrain.add(merged.get(k));
										
										List<double[]> samplesVal = new ArrayList<double[]>();
										for (int k : cvEntry.getValue()) 
											samplesVal.add(merged.get(k));
										
										List<Set<double[]>> l = new ArrayList<>();
										l.add( new HashSet<>(samplesTrain));
										PiecewiseLM plm = new PiecewiseLM(samplesTrain, l, "", fa, ta);
										
										List<Double> pred = plm.getPredictions(samplesVal, fa);
										List<Double> err = new ArrayList<>();
										for( int k = 0; k < samplesVal.size(); k++ )
											err.add( pred.get(k) - samplesVal.get(k)[ta] );
										ssm.addValue( ChowClustering.getSumOfSquares(err) );
									}
									cost = ssm.getMean() - ( ss1.getMean() + ss2.getMean() );										
								} else {
									double[] s = testStructChange(getX(s1, fa, true), getY(s1, ta), getX(s2, fa, true), getY(s2, ta), sctm);
									if( s[1] <= pValue )
										cost = s[0];
								}
								
								if( Double.isInfinite(cost) || Double.isNaN(cost) ) 
									throw new RuntimeException(cost+"");
																																
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
			s += d*d;
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

	public static double[][] getX(List<double[]> samples, int[] fa, boolean addIntercept) {
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			
			x[i] = new double[fa.length + (addIntercept ? 1 : 0) ];
			x[i][x[i].length - 1] = 1.0; // gets overwritten if !addIntercept
			for (int j = 0; j < fa.length; j++)
				x[i][j] = d[fa[j]];			
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
				System.err.println(LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS);
				System.exit(1);
				throw new MathIllegalArgumentException(LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS, x.length, x[0].length);
			}
		}
	}
	
	public static double[] testStructChange(double[][] x1, double[] y1, double[][] x2, double[] y2, StructChangeTestMode sctm) {
		int T1 = x1.length;
		int T2 = x2.length;
		int T = T1 + T2;
		int k = x1[0].length; // == x2[0].length

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
		
		if( sctm == StructChangeTestMode.ResiChow || sctm == StructChangeTestMode.ResiLikelihoodRatio || sctm == StructChangeTestMode.ResiSimple || sctm == StructChangeTestMode.ResiAICc_GWModel || sctm == StructChangeTestMode.ResiAICc 
				|| sctm == StructChangeTestMode.ResiAIC_GWModel || sctm == StructChangeTestMode.ResiAIC || sctm == StructChangeTestMode.F_TEST ) {
						
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
			} else if( sctm == StructChangeTestMode.ResiAIC_GWModel ) {
				return new double[]{ SupervisedUtils.getAIC_GWMODEL(sc/T, k, T) - SupervisedUtils.getAIC_GWMODEL( (s1+s2)/T , 2*k, T), 0 };
			} else if( sctm == StructChangeTestMode.ResiAIC ) {
				return new double[]{ SupervisedUtils.getAIC(sc/T, k, T) - SupervisedUtils.getAIC( (s1+s2)/T , 2*k, T), 0 };
			} else if( sctm == StructChangeTestMode.ResiAICc_GWModel ) {
				return new double[]{ SupervisedUtils.getAICc_GWMODEL(sc/T, k, T) - SupervisedUtils.getAICc_GWMODEL( (s1+s2)/T , 2*k, T), 0 };
			} else if( sctm == StructChangeTestMode.ResiAICc ) {
				return new double[]{ SupervisedUtils.getAICc(sc/T, k, T) - SupervisedUtils.getAICc( (s1+s2)/T , 2*k, T), 0 };
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
		
	public static int minClusterSize( Collection<Set<double[]>> ct ) {
		int min = Integer.MAX_VALUE;
		for( Set<double[]> s : ct )
			min = Math.min(s.size(), min);
		return min;
	}
}

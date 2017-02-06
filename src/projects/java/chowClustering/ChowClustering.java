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
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
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
import org.apache.commons.math3.stat.correlation.KendallsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

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
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils.ColorClass;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.GraphUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering {

	private static Logger log = Logger.getLogger(ChowClustering.class);

	public enum StructChangeTestMode {
		Chow, AdjustedChow, Wald, ResiChow, LogLikelihood, ResiSimple
	};
	
	enum PreCluster {
		Kmeans, Ward
	}
	
	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6, RUNS = 7;
	
	static LinearModel best = null;
	static Double bestAICc = Double.POSITIVE_INFINITY;
			
	public static void main(String[] args) {
				
		int threads = Math.max(1 , Runtime.getRuntime().availableProcessors() -1 );
		log.debug("Threads: "+threads);

		SpatialDataFrame sdfGWR = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden_gs2010/output/gwr_results.shp"), new int[]{ 1, 2 }, true);
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden_gs2010/gem_dat.shp"), new int[]{ 1, 2 }, true);
				
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		
		int[] ga = new int[] { 3, 4 };
		int[] fa =     new int[] {  7,  8,  9, 10, 19, 20 };
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
				
		for( int i : new int[]{ 300,400,500,600,700,800,900,1000 } ) {
			for( double p : new double[]{0.05} ) {
				params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.Wald, p, gDist, fa.length+2, PreCluster.Kmeans, i, runs });
				//params.add(new Object[] { HierarchicalClusteringType.ward, StructChangeTestMode.Chow, p, gDist, fa.length+2, PreCluster.Kmeans, i, runs });
			}
		}
																		
		Map<Object[], LinearModel> re = new HashMap<>();

		log.debug("samples: "+samples.size()+", params: "+params.size());
		for (Object[] param : params) {		
			Clustering.r.setSeed(0);
			int maxRuns = (int) param[RUNS];

			String method = Arrays.toString(param);
			int idx = params.indexOf(param);
			log.debug(idx+","+method);
			
			best = null;

			List<Future<LinearModel>> futures = new ArrayList<>();
			ExecutorService es = Executors.newFixedThreadPool(threads);
			
			for (int r = 0; r < maxRuns; r++) {
				log.debug("run "+r);
				
				List<Set<double[]>> init = getInitCluster(samples, cma, (PreCluster)param[PRECLUST], (int) param[PRECLUST_OPT], gDist );
					
				List<TreeNode> curLayer = new ArrayList<>();
				for (Set<double[]> s : init) 
					curLayer.add(new TreeNode(0, 0, s));
								
				// HC 1, maintain minobs
				{
					log.debug("hc1");
					Map<TreeNode, Set<TreeNode>> ncm = getCMforCurLayer(curLayer, cma);
					List<TreeNode> tree = Clustering.getHierarchicalClusterTree(curLayer, ncm, gDist, HierarchicalClusteringType.ward, (int) param[MIN_OBS], threads );
					curLayer = Clustering.cutTree(tree, 1);
				}
				
				// HC 2
				log.debug("hc2");											
				double pValue = (double)param[P_VALUE];
				
				// update curLayer/ncm
				for( TreeNode tn : curLayer )
					tn.contents = Clustering.getContents(tn);
				Map<TreeNode, Set<TreeNode>> ncm = getCMforCurLayer(curLayer, cma);
				
				List<TreeNode> tree = getFunctionalClusterinTree(curLayer, ncm, fa, ta, (HierarchicalClusteringType) param[CLUST], (StructChangeTestMode) param[STRUCT_TEST], pValue, threads);
								
				int minClust = Clustering.getRoots(tree).size();
				for (int i = minClust; i <= (pValue == 1.0 ? Math.min( curLayer.size(), 250) : minClust); i++ ) {
					final int nrCluster = i;
					futures.add(es.submit(new Callable<LinearModel>() {
						@Override
						public LinearModel call() throws Exception {
							List<Set<double[]>> ct = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster) );	
														
							LinearModel cr = new LinearModel(samples, ct, fa, ta, false );
							double aicc = -1; // TODO
														
							Map<double[], Double> values = new HashMap<>();
							for (int i = 0; i < samples.size(); i++)
								values.put(samples.get(i), cr.getResiduals().get(i));
							double moran = GeoUtils.getMoransI(wcm2, values);
																	
							double famh = 0;
							{
								SummaryStatistics ss = new SummaryStatistics();
								for( int i = 0; i < nrCluster; i++ ) 
									ss.addValue(cr.getBeta(i)[0]); // famH
								famh = ss.getMax()-ss.getMin();
							}
														
							double gwrCorPpDns, gwrCorFamH;
							{
								double[] xArrayPpDns = new double[samples.size()];
								double[] xArrayFamH = new double[samples.size()];
								
								double[] yArrayPpDns = new double[samples.size()];
								double[] yArrayFamH = new double[samples.size()];
								for( int i = 0; i < samples.size(); i++ ) {
									xArrayPpDns[i] = sdfGWR.samples.get(i)[3];
									xArrayFamH[i] = sdfGWR.samples.get(i)[2];
									
									double[] d = samples.get(i);
									for (int j = 0; j < cr.cluster.size(); j++) {
										if ( cr.cluster.get(j).contains(d) ) {
											yArrayPpDns[i] = cr.getBeta(j)[5]; // pop dns coef
											yArrayFamH[i] = cr.getBeta(j)[0];
											break;
										}
									}				
								}
								gwrCorPpDns = new KendallsCorrelation().correlation(xArrayPpDns, yArrayPpDns);
								gwrCorFamH = new KendallsCorrelation().correlation(xArrayFamH, yArrayFamH);
							}
							
							double[] xArraySize = new double[cr.cluster.size()];
							double[] yArrayRMSD = new double[cr.cluster.size()];
							for( int i = 0; i < cr.cluster.size(); i++ ) {
								xArraySize[i] = cr.cluster.get(i).size();
								
								List<double[]> l = new ArrayList<>(cr.cluster.get(i));
								List<Set<double[]>> c = new ArrayList<>();
								c.add(cr.cluster.get(i));
								LinearModel subLM = new LinearModel(l, c, fa, ta, false);
								yArrayRMSD[i] = Math.sqrt( subLM.getRSS()/l.size() );
							}
							double corRMSD = new KendallsCorrelation().correlation(xArraySize, yArrayRMSD);
							
							DescriptiveStatistics dsRMSE = new DescriptiveStatistics();
							{
								List<double[]> ns = new ArrayList<double[]>();
								for( Set<double[]> s : cr.cluster ) {
									int i = cr.cluster.indexOf(s);
									for( double[] d : s ) {
										double[] nd = Arrays.copyOf(d, d.length+cr.cluster.size()-1);
										if( i < cr.cluster.size() - 1 )
											nd[d.length+i] = 1;
										ns.add( nd );
									}
								}
								
								int[] nfa = Arrays.copyOf(fa, fa.length+cr.cluster.size()-1);
								for( int i = 0; i < cr.cluster.size()-1; i++ )
									nfa[fa.length+i] = samples.get(0).length+i;
																
								List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 1, ns.size());
								for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
									
									List<double[]> samplesTrain = new ArrayList<double[]>();
									for (int k : cvEntry.getKey()) 
										samplesTrain.add(ns.get(k));
																	
									DoubleMatrix X = new DoubleMatrix( LinearModel.getX( samplesTrain, nfa, true) );											
									DoubleMatrix Y = new DoubleMatrix( LinearModel.getY( samplesTrain, ta) );
									DoubleMatrix Xt = X.transpose();
									DoubleMatrix XtX = Xt.mmul(X);									
									DoubleMatrix beta = Solve.solve(XtX, Xt.mmul(Y));
									
									List<double[]> samplesVal = new ArrayList<double[]>();
									for (int k : cvEntry.getValue()) 
										samplesVal.add(ns.get(k));
									
									List<Double> predictions = new ArrayList<>();								
									DoubleMatrix PX = new DoubleMatrix( LinearModel.getX( samplesVal, nfa, true) );
									double[] p = PX.mmul(beta).data;
									for( int i = 0; i < p.length; i++ )
										predictions.add(p[i]);			
									
									dsRMSE.addValue( Math.sqrt( SupervisedUtils.getMSE(predictions, samplesVal, ta) ) );
								}
							}
							
							synchronized(this) {
								if( best == null || aicc < bestAICc ) {
									best = cr;
									bestAICc = aicc;
								}
								
								try {
									String s = "";
									s += idx + ",\"" + method + "\"," + cr.cluster.size() +","+ aicc + ","+moran+","+famh+","+gwrCorFamH+","+gwrCorPpDns+","+corRMSD+","+dsRMSE.getMean()+"\r\n";
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
			re.put(param, best);
			System.gc();			
		}
		
		// process best results of each method
		LinearModel best = null;
		for (Entry<Object[], LinearModel> e : re.entrySet()) {
			Object[] p = e.getKey();
			int idx = params.indexOf(p);
			LinearModel cr = e.getValue();
						
			double aicc = -1; // TODO
			log.info("####### "+params.indexOf(p)+" "+Arrays.toString(p)+" ########");
			log.info("#cluster: " + cr.cluster.size());
			log.info("rss: " + cr.getRSS());
			log.info("aicc: " + aicc);
			log.info("r2: " + cr.getR2(samples) );
			
			/*if( best == null || aicc < best.getAICc() )
				best = cr;*/
									
			Map<double[], Double> values = new HashMap<>();
			for (int i = 0; i < samples.size(); i++)
				values.put(samples.get(i), cr.getResiduals().get(i));
			log.info("moran: " + Arrays.toString( GeoUtils.getMoransIStatistics(wcm2, values)));
			
			{
				SummaryStatistics ss = new SummaryStatistics();
				for( int i = 0; i < cr.cluster.size(); i++ ) 
					ss.addValue(cr.getBeta(i)[0]);
				log.info("famH span: "+(ss.getMax()-ss.getMin()));
			}
			
			{
				SummaryStatistics ss = new SummaryStatistics();
				for( int i = 0; i < cr.cluster.size(); i++ ) 
					ss.addValue(cr.getBeta(i)[cr.getBeta(i).length-1]);
				log.info("Intrcpt span: "+( ss.getMax() - ss.getMin() ) );
			}
			
			List<Double> predictions = cr.getPredictions(samples, faPred);
			List<double[]> l = new ArrayList<double[]>();
			for (double[] d : samples) {				
				double[] ns = new double[3 + fa.length + 1];

				int i = samples.indexOf(d);
				ns[0] = cr.getResiduals().get(i);
				ns[1] = predictions.get(i);
				
				for (int j = 0; j < cr.cluster.size(); j++) {
					if ( !cr.cluster.get(j).contains(d) ) 
						continue;
					
					ns[2] = j;  // cluster
						
					double[] beta = cr.getBeta(j);
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
			
			DataUtils.writeShape(l, geoms, names, sdf.crs, "output/" + params.indexOf(p) + ".shp");
			
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
				
				LinearModel plm = new LinearModel( li, c, fa, ta, false);
				
				List<Double> dl = new ArrayList<>();		
				for( double d : plm.getBeta(0) )
					dl.add(d);
								
				double sss = plm.getRSS();			
				dl.add( (double)s.size() );
				dl.add( (double)cr.cluster.indexOf(s) );
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
			DataUtils.writeShape( dissSamples, dissGeoms, dissNames.toArray(new String[]{}), sdf.crs, "output/" + idx + "_diss.shp" );	
			Drawer.geoDrawValues(dissGeoms,dissSamples,fa.length+3,sdf.crs,ColorBrewer.Set3,ColorClass.Equal,"output/" + idx + "_cluster.png");
		}
	}

	public static List<TreeNode> getFunctionalClusterinTree(List<TreeNode> leafLayer, Map<TreeNode, Set<TreeNode>> cm, int[] fa, int ta, HierarchicalClusteringType hct, StructChangeTestMode sctm, double pValue, int threads) {

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
		
		Map<TreeNode, Double> ssCache = new HashMap<TreeNode, Double>();
		for( Entry<TreeNode,Set<double[]>> e : curLayer.entrySet() ) {
			List<Set<double[]>> sc1 = new ArrayList<>();
			sc1.add( e.getValue() );
			double rss1 = new LinearModel(new ArrayList<>(e.getValue()), sc1, fa, ta, false).getRSS();
			ssCache.put(e.getKey(),rss1);
		}
		Map<TreeNode, Map<TreeNode,Double>> unionCache = new ConcurrentHashMap<>();
		
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

								Set<double[]> s1 = curLayer.get(l1);
								Set<double[]> s2 = curLayer.get(l2);
								
								double cost = Double.NaN;
								if( sctm == StructChangeTestMode.ResiSimple ) { // here because we also want ridge regression work																		
									if( !unionCache.containsKey(l1) || !unionCache.get(l1).containsKey(l2) ) {
										List<double[]> l = new ArrayList<>();
										l.addAll(s1);
										l.addAll(s2);
										
										List<Set<double[]>> sc3 = new ArrayList<>();
										Set<double[]> s = new HashSet<>();
										s.addAll(s1);
										s.addAll(s2);
										sc3.add( s );
										
										double rssFull = new LinearModel(l, sc3, fa, ta, false).getRSS();
										if (!unionCache.containsKey(l1))
											unionCache.put( l1, new HashMap<TreeNode, Double>() );
										unionCache.get(l1).put(l2, rssFull);
									}					

									cost = unionCache.get(l1).get(l2) - (ssCache.get(l1) + ssCache.get(l2));
									
								} else {
									List<double[]> sl1 = new ArrayList<>(s1);
									List<double[]> sl2 = new ArrayList<>(s2);
									double[] s = testStructChange( LinearModel.getX(sl1, fa, true), LinearModel.getY(sl1, ta), LinearModel.getX(sl2, fa, true), LinearModel.getY(sl2, ta), sctm);
									if( s[1] <= pValue )
										cost = s[0];
									/*else
										cost = s[0] + 1000000;*/
								}																							
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
			
			if( sctm == StructChangeTestMode.ResiSimple ) {
				ssCache.remove(c1);
				ssCache.remove(c2);
				ssCache.put( mergeNode, unionCache.get(c1).get(c2) );
				unionCache.remove(c1);
			} 

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

	@Deprecated
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
	
	// TODO: use jblas
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
		
		if( sctm == StructChangeTestMode.ResiChow || sctm == StructChangeTestMode.LogLikelihood || sctm == StructChangeTestMode.ResiSimple ) {
						
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
			
			if( sctm == StructChangeTestMode.ResiChow ) { // F-Test
				double t = ((sc - (s1 + s2)) / k) / ((s1 + s2) / (T - 2 * k));
				FDistribution d = new FDistribution(k, T - 2 * k);				
				return new double[] { t, 1 - d.cumulativeProbability(t) }; // p-Value < 0.5 H0(equivalence) rejected, A and B not equal
			} else if(sctm == StructChangeTestMode.LogLikelihood) { // does not work
				double t = 2 * ( Math.log(s1+s2) - Math.log(sc) );
				ChiSquaredDistribution d = new ChiSquaredDistribution( k );
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
						
			return new double[]{ t, 1 - new FDistribution(k,T - 2 * k).cumulativeProbability(t) };
			
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
	
	public static Map<TreeNode, Set<TreeNode>> getCMforCurLayer( Collection<TreeNode> curLayer, Map<double[],Set<double[]>> cma ) {
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
		return ncm;
	}
	
	public static List<Set<double[]>> getInitCluster( List<double[]> samples, Map<double[],Set<double[]>> cma, PreCluster pc, int pcOpt, Dist<double[]> dist ) {
		List<Set<double[]>> init = null;
		if (pc != null && pc == PreCluster.Kmeans ) {
			List<Set<double[]>> l = new ArrayList<>(Clustering.kMeans(samples, pcOpt, dist ).values());
			init = new ArrayList<>();
			for( Set<double[]> s : l )
				if( s.isEmpty() )
					log.warn("Removing empty init cluster!");
				else
					init.add(s);					
		} else if (pc != null && pc == PreCluster.Ward ) {
			List<TreeNode> tree = Clustering.getHierarchicalClusterTree( cma, dist, HierarchicalClusteringType.ward  );
			int nrCluster = samples.size();
			do 
				init = Clustering.treeToCluster( Clustering.cutTree(tree, nrCluster--) );
			while( minClusterSize(init) <= pcOpt );
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
		return init;
	}
	
	
	public static class ValSet {
		List<double[]> samplesTrain, samplesVal;
		Map<double[],Set<double[]>> cmTrain;
	}
	
	private static Random r = new Random(10);
	
	public static ValSet getValSet( Map<double[],Set<double[]>> cm, double p) {
		Map<double[],Set<double[]>> cmTrain = new HashMap<>();
		for( Entry<double[],Set<double[]>> e : cm.entrySet() )
			cmTrain.put( e.getKey(), new HashSet<double[]>(e.getValue()));
		
		List<double[]> samplesVal = new ArrayList<>();

		while( true ) {
			List<double[]> keys = new ArrayList<>(cmTrain.keySet());
			double[] d = keys.get(r.nextInt(keys.size()));

			// no samples which have neighbors in samplesVal
			boolean valid = true;
			for( double[] nb : cm.get(d) )
				if( samplesVal.contains(nb) ) {
					valid = false;
					break;
				}
			if( !valid )
				continue;

			// copy
			Map<double[],Set<double[]>> cmTmp = new HashMap<>();
			for( Entry<double[],Set<double[]>> e : cmTrain.entrySet() )
				cmTmp.put( e.getKey(), new HashSet<double[]>(e.getValue()));

			cmTrain.remove(d);
			for( Set<double[]> s : cmTrain.values() )
				s.remove(d);

			// no samples which cut graph into subgraphs
			if( GraphUtils.getSubGraphs(cmTrain).size() > 1  ) {
				cmTrain = cmTmp;
				continue;
			}

			samplesVal.add(d);

			if( (double)samplesVal.size()/cmTrain.size() >= p ) // % val samples
				break;
		}	
		
		ValSet vs = new ValSet();
		vs.samplesTrain = new ArrayList<>(cmTrain.keySet());
		vs.samplesVal = samplesVal;
		vs.cmTrain = cmTrain;
		return vs;
	}
}

package chowClustering;

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
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.GeoUtils;
import spawnn.utils.GraphUtils;
import spawnn.utils.RegionUtils;

public class ChowClustering {

	private static Logger log = Logger.getLogger(ChowClustering.class);

	public enum StructChangeTestMode {
		Chow, AdjustedChow, Wald, ResiChow, LogLikelihood, ResiSimple
	};
	
	enum PreCluster {
		kmeans, single_linkage, complete_linkage, average_linkage, ward
	}
	
	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6, RUNS = 7;

	public static List<TreeNode> getFunctionalClusterinTree(List<TreeNode> leafLayer, Map<TreeNode, Set<TreeNode>> cm, int[] fa, int ta, StructChangeTestMode sctm, double pValue, int threads) {

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

		Map<TreeNode, Set<double[]>> curLayer = new HashMap<>();
		int age = 0;
		for (TreeNode tn : leafLayer) {
			age = Math.max(age, tn.age);
			
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
				return new ArrayList<>( curLayer.keySet() );
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
		return new ArrayList<>(curLayer.keySet());
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
		Map<TreeNode,Set<double[]>> cont = new HashMap<>();
		for( TreeNode tn : curLayer )
			cont.put(tn, Clustering.getContents(tn));
				
		Map<TreeNode, Set<TreeNode>> ncm = new HashMap<>();
		for (TreeNode tnA : curLayer) {
			Set<TreeNode> s = new HashSet<>();
			for (double[] a : cont.get(tnA) )
				for (double[] nb : cma.get(a))
					for (TreeNode tnB : curLayer)
						if (cont.get(tnB).contains(nb))
							s.add(tnB);
			ncm.put(tnA, s);
		}
		return ncm;
	}
	
	public static List<TreeNode> getInitCluster( List<double[]> samples, Map<double[],Set<double[]>> cma, PreCluster pc, int pcOpt, Dist<double[]> dist, int minObs, int threads  ) {				
		if ( pc == PreCluster.kmeans ) {
			// k-means
			List<Set<double[]>> l = new ArrayList<>(Clustering.kMeans(samples, pcOpt, dist, 0.000001 ).values());
			List<Set<double[]>> init  = new ArrayList<>();
			for( Set<double[]> s : l )
				if( s.isEmpty() )
					log.warn("Removing empty init cluster!");
				else
					init.add(s);	
			
			// to spatial contiguos cluster
			List<Set<double[]>> cInit = new ArrayList<>();
			for( Set<double[]> s : init ) 
				if( !GeoUtils.isContiugous(cma, s) ) 
					cInit.addAll( RegionUtils.getAllContiguousSubcluster(cma, s) );
				else
					cInit.add(s);
			if( cInit.size() != init.size() ) {
				//log.warn(cInit.size()+" contiguous instead of "+init.size()+" clusters");
				init = cInit;
			}
					
			// maintain min obs 
			List<TreeNode> curLayer = new ArrayList<>();
			for (Set<double[]> s : init)
				curLayer.add(new TreeNode(0, 0, s));
			Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(curLayer, cma);
					
			List<TreeNode> minObsTree = Clustering.getHierarchicalClusterTree(curLayer, ncm, dist, HierarchicalClusteringType.ward, minObs, threads);
					
			return minObsTree;
						
		} else { 
			Map<TreeNode,Set<TreeNode>> cm = Clustering.samplesCMtoTreeCM(cma);
			List<TreeNode> l = new ArrayList<>( GraphUtils.getNodes(cm) );
			if( pc == PreCluster.average_linkage )
				return Clustering.getHierarchicalClusterTree( l, cm, dist, HierarchicalClusteringType.average_linkage, minObs, threads );
			else if( pc == PreCluster.complete_linkage )
				return Clustering.getHierarchicalClusterTree( l, cm, dist, HierarchicalClusteringType.complete_linkage, minObs, threads );
			else if( pc == PreCluster.single_linkage )
				return Clustering.getHierarchicalClusterTree( l, cm, dist, HierarchicalClusteringType.single_linkage, minObs, threads );
			else if( pc == PreCluster.ward )
				return Clustering.getHierarchicalClusterTree( l, cm, dist, HierarchicalClusteringType.ward, minObs, threads );
		} 
		return null;		
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
	
	public static Map<double[], Set<double[]>> kMeans(List<double[]> samples, int num, Dist<double[]> dist, int minObs ) {
		int length = samples.iterator().next().length;
			
		Map<double[], Set<double[]>> clusters = null;
		Set<double[]> centroids = new HashSet<double[]>();

		// get num unique(!) indices for centroids
		Set<Integer> indices = new HashSet<Integer>();
		while (indices.size() < num)
			indices.add(r.nextInt(samples.size()));

		for (int i : indices) {
			double[] d = samples.get(i);
			centroids.add(Arrays.copyOf(d, d.length));
		}

		int j = 0;
		boolean changed;
		do {
			clusters = new HashMap<double[], Set<double[]>>();
			for (double[] v : centroids)
				// init cluster
				clusters.put(v, new HashSet<double[]>());

			for (double[] s : samples) { // build cluster
				double[] nearest = null;
				double nearestCost = Double.MAX_VALUE;
				int nearestSize = -1;
				
				for (double[] c : clusters.keySet()) { // find nearest centroid
					double actCost = dist.dist(c, s);
					int actSize = clusters.get(c).size();
					
					if ( nearest == null || 
							( actSize < minObs && nearestSize >= minObs ) || // too small clusters fist
							( actSize < minObs && nearestSize < minObs && actCost < nearestCost ) ||  
							( actSize >= minObs && nearestSize >= minObs && actCost < nearestCost ) ) 
					{							
						nearestSize = actSize;
						nearestCost = actCost;
						nearest = c;
					}
				}
				clusters.get(nearest).add(s);
			}

			changed = false;
			for (double[] c : clusters.keySet()) {
				Collection<double[]> s = clusters.get(c);

				if (s.isEmpty())
					continue;

				// calculate new centroids
				double[] centroid = new double[length];
				for (double[] v : s)
					for (int i = 0; i < v.length; i++)
						centroid[i] += v[i];
				for (int i = 0; i < centroid.length; i++)
					centroid[i] /= s.size();

				// update centroids
				for (int i = 0; i < c.length; i++) {
					if (c[i] != centroid[i]) {
						centroids.remove(c);
						centroids.add(centroid);
						changed = true;
					}
				}
			}

			j++;
		} while (changed && j < 1000);
		return clusters;
	}
}

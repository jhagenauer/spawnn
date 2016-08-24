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
import java.util.Collections;
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

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering {

	private static Logger log = Logger.getLogger(ChowClustering.class);
	
	static boolean debug = false;
	
	static List<TreeNode> getHierarchicalClusterTree(List<TreeNode> leafLayer, Map<TreeNode, Set<TreeNode>> cm, int[] fa, int ta, HierarchicalClusteringType hct, Dist<double[]> dist, int minSize, StructChangeTestMode sctm, int threads  ) {

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

		Map<TreeNode, Double> ssCache = new ConcurrentHashMap<TreeNode, Double>();
		Map<TreeNode, Map<TreeNode,Double>> unionCache = new ConcurrentHashMap<>();

		int age = 0;
		for (TreeNode tn : leafLayer) {
			age = Math.max(age, tn.age);
			tree.add(tn);

			Set<double[]> content = Clustering.getContents(tn);
			curLayer.put(tn, content);
			
			ssCache.put(tn, DataUtils.getSumOfSquares(content, dist));
		}
		
		List<double[]> samples = new ArrayList<double[]>();
		for( Set<double[]> s : curLayer.values() )
			samples.addAll(s);
		Collections.shuffle(samples);
				
		// copy of connected map
		final Map<TreeNode, Set<TreeNode>> connected = new HashMap<TreeNode, Set<TreeNode>>();
		if (cm != null)
			for (Entry<TreeNode, Set<TreeNode>> e : cm.entrySet())
				connected.put(e.getKey(), new HashSet<TreeNode>(e.getValue()));
		
		boolean firstPhase = true;

		while (curLayer.size() > 1) {
			
			TreeNode c1 = null, c2 = null;
			double sMin = Double.POSITIVE_INFINITY;

			List<TreeNode> cl = new ArrayList<>(curLayer.keySet());
			
			if( firstPhase) {
				boolean b = true;
				for( Set<double[]> s : curLayer.values() )
					if( s.size() < minSize )
						b = false;
				if( b ) {
					//log.debug("start second/chow phase with "+curLayer.size());
					ssCache.clear();
					unionCache.clear();
					firstPhase = false;
				}
			}
			final boolean SEC_PHASE = !firstPhase;
			
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

							for (int j = i + 1; j < cl.size(); j++) {
								TreeNode l2 = cl.get(j);

								if (!connected.containsKey(l1) || !connected.get(l1).contains(l2)) // disjoint
									continue;
								
								double cost = Double.NaN;
								List<double[]> s1 = new ArrayList<>(curLayer.get(l1));
								List<double[]> s2 = new ArrayList<>(curLayer.get(l2));
																								
								if( SEC_PHASE ) {
																																				
									Set<double[]> union = new HashSet<>(s1);
									union.addAll(s2);
														
									/*double[] chow = chowTest( 
											getSumOfSquares(getResidualsLM(null, new ArrayList<>(union), fa, ta)), 
											getSumOfSquares(getResidualsLM(null, s1, fa, ta)), 
											getSumOfSquares(getResidualsLM(null, s2, fa, ta)),  
											s1.size(), s2.size(), fa.length+1);*/
									
									cost = testStructChange(getX(s1, fa, true), getY(s1, ta), getX(s2, fa, true), getY(s2, ta), sctm );											
								} else {
									
									if( hct == HierarchicalClusteringType.ward ) {
										if( !unionCache.containsKey(l1) || !unionCache.get(l1).containsKey(l2) ) {									
											if (!unionCache.containsKey(l1))
												unionCache.put( l1, new HashMap<TreeNode, Double>() );
											
											Set<double[]> union = new HashSet<>(s1);
											union.addAll(s2);
											unionCache.get(l1).put(l2, DataUtils.getSumOfSquares(union, dist));
										}
										cost = unionCache.get(l1).get(l2) - ( ssCache.get(l1) + ssCache.get(l2) );
										
									}  else if (HierarchicalClusteringType.single_linkage == hct) {
										cost = Double.MAX_VALUE;
										for (double[] d1 : s1) 
											for (double[] d2 : s2) 
												cost = Math.min(cost, dist.dist(d1, d2) );				
									} else if (HierarchicalClusteringType.complete_linkage == hct) {
										cost = Double.MIN_VALUE;
										for (double[] d1 : s1)
											for (double[] d2 : s2)
												cost = Math.max(cost, dist.dist(d1, d2) );
									} else if (HierarchicalClusteringType.average_linkage == hct) {
										cost = 0;
										for (double[] d1 : s1) 
											for (double[] d2 : s2) 
												cost += dist.dist(d1, d2);
										cost /= (curLayer.get(l1).size() * curLayer.get(l2).size());
									}	
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
			try {
				for (Future<double[]> f : futures) {
					double[] d = f.get();
										
					if (d[2] < sMin) {
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

			if (c1 == null && c2 == null) { // no connected clusters present anymore
				log.debug("only non-connected clusters present! " + curLayer.size());
				return tree;
			}

			// create merge node, remove c1,c2
			Set<double[]> union = new FlatSet<double[]>();
			union.addAll(curLayer.remove(c1));
			union.addAll(curLayer.remove(c2));

			TreeNode mergeNode = new TreeNode(++age,sMin);
			mergeNode.children = Arrays.asList(new TreeNode[] { c1, c2 });
			ssCache.remove(c1);
			ssCache.remove(c2);

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
			ssCache.remove(c1);
			ssCache.remove(c2);
			if( hct == HierarchicalClusteringType.ward && !SEC_PHASE ) {
				ssCache.put(mergeNode,unionCache.get(c1).get(c2) );
			}
		}
		return tree;
	}

	static double getSumOfSquares(List<Double> residuals ) {
		double s = 0;
		for( double d : residuals )
			s += Math.pow(d, 2);
		return s;
	}
	
	static double getR2(double ssRes, List<double[]> samples, int ta ) {		
		SummaryStatistics ss = new SummaryStatistics();
		for (double[] d : samples)
			ss.addValue(d[ta]);
		
		double mean = 0;
		for (double[] d : samples)
			mean += d[ta];
		mean /= samples.size();

		double ssTot = 0;
		for (double[] d : samples )
			ssTot += Math.pow(d[ta] - mean, 2);
		
		return 1.0 - ssRes / ssTot;
		
	}
	
	private static double[] chowTest(double sc, double s1, double s2, int n1, int n2, int k) {
		double t = ((sc - (s1 + s2)) / k) / ( (s1 + s2) / (n1 + n2 - 2 * k) );
		
		FDistribution fd = new FDistribution(k, n1 + n2 - 2 * k);
		return new double[] { t, 1 - fd.cumulativeProbability(t) // p-Value < 0.5 H0(equivalence) rejected, A and B not equal
		};
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
			
	public static double[][] getX( List<double[]> samples, int[] fa, boolean addIntercept ) {
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			double[] stripped = getStripped(d, fa);
			if( addIntercept ) {
				x[i] = new double[stripped.length+1];
				x[i][0] = 1;
				for( int j = 0; j < stripped.length; j++ )
					x[i][j+1] = stripped[j];
			} else 
				x[i] = stripped;
		}
		return x;
	}
	
	public static double[] getY( List<double[]> samples, int ta ) {
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
	        if (x.length == 0) {  // Must be no y data either
	            throw new NoDataException();
	        }
	        if ( (isNoIntercept() ? x[0].length : x[0].length + 1) > x.length) {
	            throw new MathIllegalArgumentException(
	                    LocalizedFormats.NOT_ENOUGH_DATA_FOR_NUMBER_OF_PREDICTORS,
	                    x.length, x[0].length);
	        }
	    }
	}
	
	enum StructChangeTestMode { Chow, Wald, AChow };
	
	public static double testStructChange(double[][] x1, double[] y1, double[][]x2, double[] y2, StructChangeTestMode sctm ) {
		double T1 = x1.length;
		double T2 = x2.length;
		double T = T1+T2;
		double k = x1[0].length;
				
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
														
		if( sctm == StructChangeTestMode.Chow ) {
			
			double s1 = 0;
			for( double d : ols1.estimateResiduals() )
				s1 += d*d;
			
			double s2 = 0;
			for( double d : ols2.estimateResiduals() )
				s2 += d*d;
			
			double c = diff.transpose().multiply( MatrixUtils.inverse( m1.add(m2) ) ).multiply(diff).getEntry(0, 0)*(T-2*k) / (k * ( s1 + s2 ) ); // basic chow
			return c;
		} else { 
			double s1 = ols1.estimateErrorVariance();
			double s2 = ols2.estimateErrorVariance();
			double w = diff.transpose().multiply( MatrixUtils.inverse( m1.scalarMultiply(s1).add( m2.scalarMultiply(s2) ) ) ).multiply(diff).getEntry(0, 0);
			if( sctm == StructChangeTestMode.Wald ) 
				return w;
			else if( sctm == StructChangeTestMode.AChow )
				return w/k;
		}
		return -1;
	}
	
	public static List<Double> getResidualsLM( List<Set<double[]>> cluster, List<double[]> samplesTrain, List<double[]> samplesVal, int[] fa, int ta ) {
		Set<double[]> all = new HashSet<double[]>();
		int sumSize = 0;
		for( Set<double[]> s : cluster ) {
			sumSize+=s.size();
			all.addAll(s);
		}
		if( sumSize != all.size() )
			throw new RuntimeException("Cluster overlap!");
			
		if( !all.containsAll(samplesTrain) || !all.containsAll(samplesVal) )
			throw new RuntimeException("Some samples not assigend to cluster!");
				
		List<Double> residuals = new ArrayList<>();
		for( Set<double[]> s : cluster ) {

			List<double[]> l = new ArrayList<>();
			for( double[] d : samplesTrain )
				if( s.contains(d ))
					l.add(d);
					
			double[] y = new double[l.size()];
			double[][] x = new double[l.size()][];
			for (int i = 0; i < l.size(); i++) {
				double[] d = l.get(i);
				y[i] = d[ta];
				x[i] = getStripped(d, fa);
			}
						
			// training
			double[] beta = null;
			try {
				OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
				ols.setNoIntercept(false);
				ols.newSampleData(y, x);
				beta = ols.estimateRegressionParameters();							
			} catch( Exception e ) {
				//System.err.println("Too few samples to learn lm, predicting mean value.");
				
				beta = new double[fa.length+1];
				for( double d : y )
					beta[0] += d;
				beta[0] /= y.length;
			}
			
			// validation
			l = new ArrayList<>();
			for( double[] d : samplesVal )
				if( s.contains(d ))
					l.add(d);
						
			for (int i = 0; i < l.size(); i++) {
				double[] d = l.get(i);
				double[] xi = getStripped(d, fa);
								
				double p = beta[0]; // intercept at beta[0]
				
				
				for (int j = 1; j < beta.length; j++)
					p += beta[j] * xi[j - 1];
						
				residuals.add( d[ta] - p );
			}
		}
						
		return residuals;
	}
	
	static class ClusterResult {
		ClusterResult(int in, double cost, List<Set<double[]>> cluster, String method ) { this.in = in; this.cost = cost; this.cluster = cluster; this.method = method; }
		String method;
		int in;
		double cost;
		List<Set<double[]>> cluster = null;
		
		@Override
		public String toString() {
			return method+","+in+","+cluster.size()+","+cost;
		}
	}
	
	public static void main(String[] args) {	
		int threads = 11;
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/gemeinden/gem_dat.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		
		Dist<double[]> gDist = new EuclideanDist(new int[] { 1, 2 });
		
		int[] fa = new int[]{ 0, 4, 5, 6, 7, 9 };
		int ta = 12;
		
		DataUtils.transform(samples, new int[]{0}, Transform.sqrt );
		DataUtils.transform(samples, new int[]{4}, Transform.log );
		
		log.debug(Arrays.toString(fa)+":"+ta);
		
		Path file = Paths.get("output/chow.txt");
		try {
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		List<Entry<List<Integer>, List<Integer>>> cvList5 = SupervisedUtils.getCVList(5, 10, samples.size());	
		List<Entry<List<Integer>, List<Integer>>> cvList10 = SupervisedUtils.getCVList(10, 10, samples.size());		
		
		// ------------------------------------------------------------------------------------------
				
		for( int in : new int[]{ -14, -12, -10, -8, 400, 500, 600, 700 })
		for( HierarchicalClusteringType hct : new HierarchicalClusteringType[]{ HierarchicalClusteringType.ward, HierarchicalClusteringType.average_linkage } )
		for( StructChangeTestMode sctm : new StructChangeTestMode[]{ /*StructChangeTestMode.Chow, StructChangeTestMode.AChow,*/ StructChangeTestMode.Wald } ) {
			Clustering.r.setSeed(0);
						
			ClusterResult bestAIC = null, bestCV5 = null, bestCV10 = null;
													
			double[][] aics;
			double[][] errorsCV5;
			double[][] errorsCV10;
			
			int maxRuns;
			if( in < 0 ) 
				maxRuns = 1;
			else
				maxRuns = 24; // 24
						
			errorsCV5 = new double[maxRuns][];
			errorsCV10 = new double[maxRuns][]; // nr errors == runs
			aics = new double[maxRuns][];
			
			String method = in+","+hct+","+sctm;
			log.debug(method);
						
			for( int r = 0; r < maxRuns; r++ ) {
				log.debug(r);
						
				List<Set<double[]>> init;
				if( in < 0 ) {
					init = new ArrayList<>();
					for( double[] d : samples ) {
						Set<double[]> s = new HashSet<double[]>();
						s.add(d);
						init.add(s);			
					}
				} else {
					init = new ArrayList<>(Clustering.kMeans(samples, in, gDist).values());
				}
										
				{
					SummaryStatistics ss = new SummaryStatistics();
					for (Set<double[]> s : init)
						ss.addValue( s.size() );
					//log.debug(ss.getMin()+","+ss.getMean()+","+ss.getMax());
				}
				
				List<TreeNode> curLayer = new ArrayList<>();
				for (Set<double[]> s : init) {
					TreeNode cn = new TreeNode(0,0);
					cn.setContents(s);
					curLayer.add(cn);
				}
				
				Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(samples, geoms, false);
				Map<TreeNode, Set<TreeNode>> ncm = new HashMap<>();
				for (TreeNode tnA : curLayer) {
					Set<TreeNode> s = new HashSet<>();
					for (double[] a : tnA.contents)
						for (double[] nb : cm.get(a))
							for (TreeNode tnB : curLayer)
								if (tnB.contents.contains(nb))
									s.add(tnB);
					ncm.put(tnA, s);
				}
				
				int m = fa.length + 2;
				if( in < 0 )
					m = Math.abs(in);
						
				List<TreeNode> tree = getHierarchicalClusterTree(curLayer, ncm, fa, ta, hct, gDist, m, sctm, threads); // 7 because error variance calculation
				
				List<Integer> nrCl = new ArrayList<>();
				for( int i = 4; i < Math.min(curLayer.size(), 200); i+= 4 )
					nrCl.add(i);				

				errorsCV5[r] = new double[nrCl.size()];
				errorsCV10[r] = new double[nrCl.size()];
				aics[r] = new double[nrCl.size()];
				for( int i = 0; i < nrCl.size(); i++ ) {
					List<Set<double[]>> ct = Clustering.cutTree(tree, nrCl.get(i) );	
					
					// CV 10
					{
					ExecutorService es = Executors.newFixedThreadPool(threads);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();	
					for( final Entry<List<Integer>,List<Integer>> cvEntry : cvList10 ) {
						futures.add(es.submit(new Callable<double[]>() {
								@Override
								public double[] call() throws Exception {
									List<double[]> samplesTrain = new ArrayList<double[]>();
									for( int k : cvEntry.getKey() )
										samplesTrain.add(samples.get(k));
												
									List<double[]> samplesVal = new ArrayList<double[]>();
									for( int k : cvEntry.getValue() )
										samplesVal.add(samples.get(k));
									
									return new double[]{ Math.sqrt( getSumOfSquares( getResidualsLM( ct, samplesTrain, samplesVal, fa, ta) )/samplesVal.size() ) };	
								}
							}));
					}
					es.shutdown();
										
					SummaryStatistics ss = new SummaryStatistics();
					for( Future<double[]> f : futures )
						try {
							ss.addValue( f.get()[0] );
						} catch (InterruptedException | ExecutionException e) {
							e.printStackTrace();
						}
					errorsCV10[r][i] = ss.getMean();
					}
					
					// CV 5
					{
					ExecutorService es = Executors.newFixedThreadPool(threads);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();	
					for( final Entry<List<Integer>,List<Integer>> cvEntry : cvList5 ) {
						futures.add(es.submit(new Callable<double[]>() {
								@Override
								public double[] call() throws Exception {
									List<double[]> samplesTrain = new ArrayList<double[]>();
									for( int k : cvEntry.getKey() )
										samplesTrain.add(samples.get(k));
												
									List<double[]> samplesVal = new ArrayList<double[]>();
									for( int k : cvEntry.getValue() )
										samplesVal.add(samples.get(k));
									
									return new double[]{ Math.sqrt( getSumOfSquares( getResidualsLM( ct, samplesTrain, samplesVal, fa, ta) )/samplesVal.size() ) };	
								}
							}));
					}
					es.shutdown();
										
					SummaryStatistics ss = new SummaryStatistics();
					for( Future<double[]> f : futures )
						try {
							ss.addValue( f.get()[0] );
						} catch (InterruptedException | ExecutionException e) {
							e.printStackTrace();
						}
					errorsCV5[r][i] = ss.getMean();
					}
					aics[r][i] = SupervisedUtils.getAICc(getSumOfSquares( getResidualsLM( ct, samples, samples, fa, ta) )/samples.size(), ct.size()*(fa.length+1), samples.size() );
					
					if( bestAIC == null || aics[r][i] < bestAIC.cost ) 
						bestAIC = new ClusterResult(in,aics[r][i], ct, "aic_"+method);
					
					if( bestCV5 == null || errorsCV5[r][i] < bestCV5.cost ) 
						bestCV5 = new ClusterResult(in,errorsCV5[r][i], ct, "cv5_"+method);
					
					if( bestCV10 == null || errorsCV10[r][i] < bestCV10.cost ) 
						bestCV10 = new ClusterResult(in,errorsCV10[r][i], ct, "cv10_"+method);
					
				}
			}
												
			try {
				String sErrorCV5 = 	"rmse_cv5,"	+method+","+Arrays.toString(getMean(errorsCV5))+"\r\n";
				String sErrorCV10 = "rmse_cv10,"+method+","+Arrays.toString(getMean(errorsCV10))+"\r\n";
				String sAIC = 		"aic,"		+method+","+Arrays.toString(getMean(aics))+"\r\n";
				
				Files.write(file, sErrorCV5.replace("[", "").replace("]", "").getBytes(), StandardOpenOption.APPEND );
				Files.write(file, sErrorCV10.replace("[", "").replace("]", "").getBytes(), StandardOpenOption.APPEND );
				Files.write(file, sAIC.replace("[", "").replace("]", "").getBytes(), StandardOpenOption.APPEND );
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			for( ClusterResult cr : new ClusterResult[]{bestAIC,bestCV5,bestCV10} ) {
				log.info(cr);
				Drawer.geoDrawCluster(cr.cluster, samples, geoms, "output/"+cr.method+"_"+cr.cluster.size()+"_"+cr.cost+".png", true);
				
				/*List<double[]> l = new ArrayList<double[]>();
				for( double[] d : samples ) {
					double[] ns = Arrays.copyOf(d, d.length+1);
					for( int i = 0; i < cr.cluster.size(); i++ ) {
						if( cr.cluster.get(i).contains(d) ) {
							ns[ns.length-1] = i;
							break;
						}
					}
					l.add(ns);
				}
				String[] names = sdf.getNames();
				names = Arrays.copyOf(names, names.length+1);
				names[names.length-1] = "cluster";
				DataUtils.writeShape(l, geoms, names, sdf.crs, "output/"+cr.method+".shp");*/
			}
		}
	}
		
	public static double[] getMean( double[][] e ) {
		double[] mean = new double[e[0].length];
		for( int r = 0; r < e.length; r++ )
			for( int i = 0; i < e[r].length; i++ )
				mean[i] += e[r][i]/e.length;
		return mean;
	}
}

package chowClustering;

import java.io.File;
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

import org.apache.commons.math3.distribution.FDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering {

	private static Logger log = Logger.getLogger(ChowClustering.class);

	static List<TreeNode> getHierarchicalClusterTree(List<TreeNode> leafLayer, Map<TreeNode, Set<TreeNode>> cm, int[] fa, int ta, int threads) {

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
			TreeNode c1 = null, c2 = null;
			double sMin = Double.MAX_VALUE;

			List<TreeNode> cl = new ArrayList<>(curLayer.keySet());

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			for (int t = 0; t < threads; t++) {
				final int T = t;

				futures.add(es.submit(new Callable<double[]>() {
					@Override
					public double[] call() throws Exception {
						int c1 = -1, c2 = -1;
						double sMin = Double.MAX_VALUE;

						for (int i = T; i < cl.size() - 1; i += threads) {
							TreeNode l1 = cl.get(i);

							for (int j = i + 1; j < cl.size(); j++) {
								TreeNode l2 = cl.get(j);

								if (!connected.containsKey(l1) || !connected.get(l1).contains(l2)) // disjoint
									continue;
								
								List<double[]> s1 = new ArrayList<>(curLayer.get(l1));
								if( !ssCache.containsKey(l1) )
									ssCache.put(l1, getSumOfSquares(getResidualsLM(null, s1, s1, fa, ta)));
																
								List<double[]> s2 = new ArrayList<>(curLayer.get(l2));
								if( !ssCache.containsKey(l2) )
									ssCache.put( l2,getSumOfSquares(getResidualsLM(null, s2, s2, fa, ta)));
								
								Set<double[]> union = new HashSet<>(s1);
								union.addAll(s2);
								List<double[]> u = new ArrayList<>(union);
								double uResi = getSumOfSquares(getResidualsLM(null, u, u, fa, ta));
								
								double s = chowTest(uResi, ssCache.get(l1), ssCache.get(l2), s1.size(), s2.size(), fa.length)[0];
								if (s < sMin) {
									c1 = i;
									c2 = j;
									sMin = s;
								}
							}
						}
						return new double[] { c1, c2, sMin };
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
			TreeNode mergeNode = new TreeNode();
			Set<double[]> union = new FlatSet<double[]>();
			union.addAll(curLayer.remove(c1));
			union.addAll(curLayer.remove(c2));
			mergeNode.age = ++age;
			mergeNode.children = Arrays.asList(new TreeNode[] { c1, c2 });
			ssCache.remove(c1);
			ssCache.remove(c2);

			// calculate/update cost
			mergeNode.cost = sMin;

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

	private static double[] chowTest(double s, double sA, double sB, int nA, int nB, int k) {
		double T = ((s - (sA + sB)) / k) / ( (sA + sB) / (nA + nB - 2 * k) );
		FDistribution fd = new FDistribution(k, nA + nB - 2 * k);
		return new double[] { T, 1 - fd.cumulativeProbability(T) // p-Value < 0.5, H0(equivalence) rejected, A and B not equal
		};
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
	
	// Yeah, yeah... not optimal, I know
	public static List<Double> getResidualsLM( List<Set<double[]>> cluster, List<double[]> samplesTrain, List<double[]> samplesVal, int[] fa, int ta ) {
		// check that all samples are assigned to a cluster
		if( cluster != null ) {
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
		}
						
		double[] y = new double[samplesTrain.size()];
		for (int i = 0; i < samplesTrain.size(); i++)
			y[i] = samplesTrain.get(i)[ta];

		double[][] x = new double[samplesTrain.size()][];
		for (int i = 0; i < samplesTrain.size(); i++) {
			double[] d = samplesTrain.get(i);
			x[i] = getStripped(d, fa);
					
			if( cluster != null ) {
				int length = x[i].length;
				x[i] = Arrays.copyOf(x[i], length + cluster.size() - 1);
				for( int idx = 0; idx < cluster.size()-1; idx++ ) {
					if( cluster.get(idx).contains(d) ) {
						x[i][length + idx] = 1;
						break;
					}
				}				
			}
		}
					
		// training
		double[] beta = null;
		try {
			OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
			ols.setNoIntercept(false);
			ols.newSampleData(y, x);
			beta = ols.estimateRegressionParameters();
						
		} catch( Exception e ) {
			e.printStackTrace();
			System.exit(1);
		}
				
		List<Double> residuals = new ArrayList<>();
		for (int i = 0; i < samplesVal.size(); i++) {
			double[] d = samplesVal.get(i);
			double[] xi = getStripped(d, fa);
			
			if( cluster != null ) {
				int length = xi.length;
				xi = Arrays.copyOf(xi, length + cluster.size() - 1);
				for( int idx = 0; idx < cluster.size(); idx++ ) {			
					if( cluster.get(idx).contains(d) ) {
						if( idx < cluster.size()-1 )
							xi[length+idx] = 1;
						break;
					}
				}
			}
			
			double p = beta[0]; // intercept at beta[0]
			for (int j = 1; j < beta.length; j++)
				p += beta[j] * xi[j - 1];
			residuals.add( samplesVal.get(i)[ta] - p );
		}
		return residuals;
	}

	public static void main(String[] args) {
		Random r = new Random();
		int threads = 3;
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/gemeinden/gem_dat.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;

		Dist<double[]> gDist = new EuclideanDist(new int[] { 1, 2 });
		
		int[] fa = new int[]{ 4,5,6,7,9};
		int ta = 12;
		DataUtils.transform(samples, new int[]{4}, Transform.log );

		Map<double[], Set<double[]>> kmCluster = Clustering.kMeans(samples, 700, gDist);
		Drawer.geoDrawCluster(kmCluster.values(), samples, geoms, "output/km_cluster.png",true);
				
		{
		SummaryStatistics ss = new SummaryStatistics();
		for (Set<double[]> s : kmCluster.values())
			ss.addValue( s.size() );
		log.debug(ss.getMin()+","+ss.getMean()+","+ss.getMax());
		}

		List<TreeNode> curLayer = new ArrayList<>();
		for (Set<double[]> s : kmCluster.values()) {
			TreeNode cn = new TreeNode();
			cn.age = 0;
			cn.cost = 0;
			cn.contents = s;
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
		
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 20, samples.size());
		
		long time = System.currentTimeMillis();
		
		log.debug("Build tree...");
		List<TreeNode> tree = getHierarchicalClusterTree(curLayer, ncm, fa, ta, threads);
		log.debug("done.");
				
		int minNr = -1;
		double minError = Double.MAX_VALUE;
		
		for( int i = 2; i < kmCluster.size(); i+= 10 ) {
			List<Set<double[]>> ct = Clustering.cutTree(tree, i);	
			
			// CV
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();	
			for( final Entry<List<Integer>,List<Integer>> cvEntry : cvList ) {
				futures.add(es.submit(new Callable<double[]>() {
						@Override
						public double[] call() throws Exception {
							List<double[]> samplesTrain = new ArrayList<double[]>();
							for( int k : cvEntry.getKey() )
								samplesTrain.add(samples.get(k));
										
							List<double[]> samplesVal = new ArrayList<double[]>();
							for( int k : cvEntry.getValue() )
								samplesVal.add(samples.get(k));
							
							double ss = getSumOfSquares( getResidualsLM( ct, samplesTrain, samplesVal, fa, ta) );
							return new double[]{ Math.sqrt(ss/samplesVal.size()) }; // RMSE	
						}
					}));
			}
			es.shutdown();
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( Future<double[]> f : futures )
				try {
					ds.addValue(f.get()[0]);
					if( f.get()[0] < minError ) {
						minNr = ct.size();
						minError = f.get()[0];
					}
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			
			double ss = getSumOfSquares( getResidualsLM( ct, samples, samples, fa, ta) );
			double rmse = Math.sqrt(ss/samples.size()); 	
						
			log.debug("Nr: " + ct.size()+", CV: "+ds.getMean()+", full: "+rmse+", R2: "+getR2(ss, samples, ta) );
		}
		log.debug("Min: "+minNr+":"+minError);
		log.debug("Took: " + (System.currentTimeMillis() - time) / 1000.0);
	}
}

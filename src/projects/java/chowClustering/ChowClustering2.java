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
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class ChowClustering2 {

	private static Logger log = Logger.getLogger(ChowClustering2.class);
	
	private static List<TreeNode> getHierarchicalClusterTreeInit(List<TreeNode> leafLayer, Map<TreeNode, Set<TreeNode>> cm, Dist<double[]> dist, int minSize, int threads) {

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

		Map<TreeNode, Double> ssCache = new HashMap<TreeNode, Double>();
		Map<TreeNode, Map<TreeNode,Double>> unionCache = new HashMap<>();

		int length = Clustering.getContents(leafLayer.get(0)).iterator().next().length;
		int age = 0;
		for (TreeNode tn : leafLayer) {
			age = Math.max(age, tn.age);
			tree.add(tn);

			Set<double[]> content = Clustering.getContents(tn);
			curLayer.put(tn, content);
			ssCache.put(tn, DataUtils.getSumOfSquares(content, dist));
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
						double bestCost = Double.MAX_VALUE;
						int bestSize = Integer.MAX_VALUE;

						for (int i = T; i < cl.size() - 1; i += threads) {
							TreeNode l1 = cl.get(i);

							for (int j = i + 1; j < cl.size(); j++) {
								TreeNode l2 = cl.get(j);

								if (!connected.containsKey(l1) || !connected.get(l1).contains(l2)) // disjoint
									continue;
																
								if (!unionCache.containsKey(l1) || !unionCache.get(l1).containsKey(l2)) {
									
									// calculate mean and sum of squares, slightly faster than actually forming a union
									double[] mean = new double[length];
									for (int l = 0; l < length; l++) {
										for (double[] d : curLayer.get(l1) )
											mean[l] += d[l];
										for (double[] d : curLayer.get(l2) )
											mean[l] += d[l];
									}
																
									for (int l = 0; l < length; l++)
										mean[l] /= curLayer.get(l1).size()+curLayer.get(l2).size();
									
									double ssUnion = 0;
									for( double[] d : curLayer.get(l1) ) {
										double di = dist.dist(mean, d);
										ssUnion += di * di;
									}
									for( double[] d : curLayer.get(l2) ) {
										double di = dist.dist(mean, d);
										ssUnion += di * di;
									}
									
									if (!unionCache.containsKey(l1))
										unionCache.put( l1, new HashMap<TreeNode, Double>() );
									unionCache.get(l1).put(l2, ssUnion);
								}													
								double cost = unionCache.get(l1).get(l2) - ( ssCache.get(l1) + ssCache.get(l2) );					
								int unionSize = curLayer.get(l1).size()+curLayer.get(l2).size();			
								
								if ( (cost < bestCost && unionSize < minSize ) // if size ok, dist matters
										|| ( bestSize > minSize && unionSize < bestSize )  
										|| ( bestSize > minSize && unionSize == bestSize  && cost < bestCost ) ) {
								/*if ( (unionSize < bestSize )	
										|| ( unionSize == bestSize && cost < bestCost ) ) {*/
									c1 = i;
									c2 = j;
									bestCost = cost;
									bestSize = unionSize;
								}
							}
						}
						return new double[] { c1, c2, bestCost };
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
			
			boolean allMinSize = true;
			for( Set<double[]> s : curLayer.values() )
				if( s.size() < minSize )
					allMinSize = false;
			if( allMinSize )
				return tree;

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
			mergeNode.cost = sMin;
						
			// update caches
			ssCache.remove(c1);
			ssCache.remove(c2);
			ssCache.put( mergeNode, unionCache.get(c1).get(c2) );
			unionCache.remove(c1);
			
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
			log.debug(curLayer.size());
		}
		return tree;
	}

	public static void main(String[] args) {
		int threads = 3;
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/gemeinden/gem_dat.shp"), true);
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;

		Dist<double[]> gDist = new EuclideanDist(new int[] { 1, 2 });
		
		int[] fa = new int[]{ 4,5,6,7,9};
		int ta = 12;
		DataUtils.transform(samples, new int[]{4}, Transform.log );
		
		Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(samples, geoms, false);
		
		Set<double[]> set = new HashSet<double[]>();
		for( Entry<double[],Set<double[]>> e : cm.entrySet() ) {
			set.add(e.getKey());
			set.addAll(e.getValue());
		}
				
		List<TreeNode> l = new ArrayList<TreeNode>();
		for( double[] d : set ) {
			TreeNode tn = new TreeNode();
			tn.age = 0;
			tn.cost = 0;
			tn.contents = new HashSet<>();
			tn.contents.add(d);
			l.add(tn);
		}

		Map<TreeNode,Set<TreeNode>> ncm = new HashMap<>();
		for( TreeNode tnA : l ) {
			double[] a = tnA.contents.iterator().next();
			
			Set<TreeNode> s = new HashSet<>();
			for( double[] b : cm.get(a) )
				for( TreeNode tnB : l )
					if( tnB.contents.contains(b) )
						s.add(tnB);
			ncm.put(tnA, s);
		}
		List<Set<double[]>> kmCluster = new ArrayList<>();
		List<TreeNode> tr = getHierarchicalClusterTreeInit(l, ncm, gDist, 12, 3);
		
		List<TreeNode> leafLayer = new ArrayList<>();
		for( TreeNode tnA : tr ) {
			boolean isChild = false;
			for( TreeNode tnB : tr )
				if( tnB.children.contains(tnA) )
					isChild = true;
			if( !isChild ) {
				leafLayer.add(tnA);
				kmCluster.add( Clustering.getContents(tnA));
			}
		}
		
		Map<TreeNode,Set<TreeNode>> llCm = new HashMap<>();
		for( TreeNode tnA : leafLayer ) {
			Set<double[]> sa = Clustering.getContents(tnA);
			
			Set<TreeNode> s = new HashSet<>();
			for( TreeNode tnB : leafLayer ) {
				if( tnA == tnB )
					continue;
				Set<double[]> sb = Clustering.getContents(tnB);
				for( double[] a : sa )
					for( double[] nb : cm.get(a) )
						if( sb.contains(nb) )
							s.add(tnB);
			}
			llCm.put(tnA, s);
		}
		
		Drawer.geoDrawCluster(kmCluster, samples, geoms, "output/hcClust.png", true);
							
		{
		SummaryStatistics ss = new SummaryStatistics();
		for (Set<double[]> s : kmCluster)
			ss.addValue( s.size() );
		log.debug(ss.getMin()+","+ss.getMean()+","+ss.getMax());
		}
		
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(10, 20, samples.size());
		
		long time = System.currentTimeMillis();
		List<TreeNode> tree = ChowClustering.getHierarchicalClusterTree(leafLayer, llCm, fa, ta, threads);
				
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
							
							double ss = ChowClustering.getSumOfSquares( ChowClustering.getResidualsLM( ct, samplesTrain, samplesVal, fa, ta) );
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
			
			double ss = ChowClustering.getSumOfSquares( ChowClustering.getResidualsLM( ct, samples, samples, fa, ta) );
			double rmse = Math.sqrt(ss/samples.size()); 	
						
			log.debug("Nr: " + ct.size()+", CV: "+ds.getMean()+", full: "+rmse+", R2: "+ChowClustering.getR2(ss, samples, ta) );
		}
		log.debug("Min: "+minNr+":"+minError);
		log.debug("Took: " + (System.currentTimeMillis() - time) / 1000.0);
	}
}

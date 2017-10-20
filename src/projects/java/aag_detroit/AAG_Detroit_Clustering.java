package aag_detroit;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import spawnn.dist.AugmentedDist;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GraphClustering;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class AAG_Detroit_Clustering {

	enum Mode {
		Augmented, Weighted, CNG
	};
	
	public static void main(String[] args) {

		Comparator<Set<double[]>> comp = new Comparator<Set<double[]>>() {
			@Override
			public int compare(Set<double[]> o1, Set<double[]> o2) {
				return Double.compare(o1.size(), o2.size());
			}
		};
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("/home/julian/publications/aag_detroit/data/detroit_metro_race.shp"), true);
		int[] fa = new int[] { 0, 1, 2, 3, 4, 5 };
		int[] ga = new int[] { 9, 10 };
		Dist<double[]> gDist = new EuclideanDist(ga);
		Dist<double[]> fDist = new EuclideanDist(fa);

		List<double[]> origSamples = new ArrayList<double[]>();
		for( double[] d : sdf.samples )
			origSamples.add( Arrays.copyOf( d, d.length ) );
		
		DataUtils.transform(sdf.samples, fa, Transform.zScore);
		DataUtils.zScoreGeoColumns(sdf.samples, ga, gDist);

		Random r = new Random();
		int T_MAX = 100000;
		int nrNeurons = 96;

		for (Mode m : new Mode[]{Mode.CNG, Mode.Weighted}) {
			r.setSeed(0);
			System.out.println(m);

			Object o;
			if (m == Mode.Augmented) {
				o = 0.9;
			} else if (m == Mode.Weighted) {
				o = 0.42;
			} else {
				o = 14;
			}
			Sorter<double[]> s = null;
			if (m == Mode.Augmented) {
				double a = (double) o;
				Dist<double[]> aDist = new AugmentedDist(ga, fa, a);
				s = new DefaultSorter<double[]>(aDist);
			} else if (m == Mode.Weighted) {
				double w = (double) o;
				Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
				map.put(fDist, 1 - w);
				map.put(gDist, w);
				Dist<double[]> wDist = new WeightedDist<double[]>(map);
				s = new DefaultSorter<double[]>(wDist);
			} else {
				s = new KangasSorter<double[]>(gDist, fDist, (int) o);
			}

			ArrayList<double[]> neurons = new ArrayList<double[]>();
			for (int j = 0; j < nrNeurons; j++) {
				double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
				neurons.add(Arrays.copyOf(d, d.length));
			}

			NG ng = new NG(neurons, nrNeurons / 2, 0.01, 0.5, 0.005, s);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = sdf.samples.get(r.nextInt(sdf.samples.size()));
				ng.train((double) t / T_MAX, x);
			}

			Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(sdf.samples, ng.getNeurons(), s);
			System.out.println("f-qe: " + DataUtils.getMeanQuantizationError(bmus, fDist));
			System.out.println("s-qe: " + DataUtils.getMeanQuantizationError(bmus, gDist));

			System.out.println("f-bmu-wcss: " + ClusterValidation.getWithinClusterSumOfSuqares(bmus.values(), fDist));
			System.out.println("s-bmu-wcss: " + ClusterValidation.getWithinClusterSumOfSuqares(bmus.values(), gDist));

			Map<double[], Map<double[], Double>> graph = new HashMap<double[], Map<double[], Double>>();
			for (double[] x : sdf.samples) {
				s.sort(x, neurons);
				double[] bmuA = neurons.get(0);
				double[] bmuB = neurons.get(1);
				if (!graph.containsKey(bmuA))
					graph.put(bmuA, new HashMap<double[], Double>());
				if (!graph.containsKey(bmuB))
					graph.put(bmuB, new HashMap<double[], Double>());

				if (!graph.get(bmuA).containsKey(bmuB))
					graph.get(bmuA).put(bmuB, 0.0);
				if (!graph.get(bmuB).containsKey(bmuA))
					graph.get(bmuB).put(bmuA, 0.0);

				graph.get(bmuA).put(bmuB, graph.get(bmuA).get(bmuB) + 1.0);
				graph.get(bmuB).put(bmuA, graph.get(bmuB).get(bmuA) + 1.0);
			}

			Map<double[], Integer> map = GraphClustering.multilevelOptimization(graph, 10);
			List<Set<double[]>> neuronClusters = new ArrayList<Set<double[]>>(GraphClustering.modulMapToCluster(map));
			Collections.sort(neuronClusters, comp);
			System.out.println("clusters: " + neuronClusters.size());

			List<Set<double[]>> finClusters = new ArrayList<Set<double[]>>();
			for (Set<double[]> s1 : neuronClusters) {
				Set<double[]> s2 = new HashSet<double[]>();
				for (double[] d : s1)
					s2.addAll(bmus.get(d));
				finClusters.add(s2);
			}
			Collections.sort(finClusters, comp);
			
			System.out.println("f-wcss: " + ClusterValidation.getWithinClusterSumOfSuqares(finClusters, fDist));
			System.out.println("s-wcss: " + ClusterValidation.getWithinClusterSumOfSuqares(finClusters, gDist));
			
			for( int i = 0; i < finClusters.size(); i++ ) {
				Set<double[]> se = finClusters.get(i);
				System.out.println("f "+i+" "+DataUtils.getSumOfSquares(se, fDist)+","+DataUtils.getSumOfSquares(se, fDist)/se.size() );
				System.out.println("s "+i+" "+DataUtils.getSumOfSquares(se, gDist)+","+DataUtils.getSumOfSquares(se, gDist)/se.size() );
			}

			Drawer.geoDrawCluster(finClusters, sdf.samples, sdf.geoms, "output/clusters_" + m + ".png", true);

			// save final cluster --------------------

			List<double[]> nSamples = new ArrayList<double[]>();
			for (double[] d : sdf.samples) {
				double[] od = origSamples.get( sdf.samples.indexOf(d) );
				
				double[] nd = new double[d.length+od.length+1];
				for( int i = 0; i < d.length; i++ )
					nd[i] = d[i];
				
				for( int i = 0; i < od.length; i++ )
					nd[d.length+i] = od[i];
				for (int i = 0; i < finClusters.size(); i++)
					if (finClusters.get(i).contains(d)) {
						nd[nd.length - 1] = i;
						break;
					}
				nSamples.add(nd);
			}

			List<String> nNames = new ArrayList<String>();
			for( String st : sdf.names )
				nNames.add(st);
			for( String st : sdf.names )
				nNames.add(st+"1");
			nNames.add("cluster");
			
			DataUtils.writeShape(nSamples, sdf.geoms, nNames.toArray(new String[]{}), sdf.crs, "output/clusters_" + m + ".shp");

			// save final graph ----------------

			Set<double[]> vertices = new HashSet<double[]>(graph.keySet());
			for (Map<double[], Double> m1 : graph.values())
				vertices.addAll(m1.keySet());

			Map<double[], double[]> nVertices = new HashMap<double[], double[]>();
			for (double[] a : graph.keySet()) {
				double[] na = Arrays.copyOf(a, a.length + 1);
				for (int i = 0; i < neuronClusters.size(); i++)
					if (neuronClusters.get(i).contains(a)) {
						na[na.length - 1] = i;
						break;
					}
				nVertices.put(a, na);
			}

			Map<double[], Map<double[], Double>> nGraph = new HashMap<double[], Map<double[], Double>>();
			for (double[] a : graph.keySet()) {
				double[] na = nVertices.get(a);
				nGraph.put(na, new HashMap<double[], Double>());

				for (double[] b : graph.get(a).keySet()) {
					double[] nb = nVertices.get(b);
					nGraph.get(na).put(nb, graph.get(a).get(b));
				}
			}
			
			nNames = new ArrayList<String>();
			nNames.addAll(sdf.names);
			nNames.add("cluster");
			
			GraphUtils.writeGraphToGraphML(nNames.toArray(new String[]{}), nGraph, new File("output/graph_" + m + ".xml"));
		}
	}
}

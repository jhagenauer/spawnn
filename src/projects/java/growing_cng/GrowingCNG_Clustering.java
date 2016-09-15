package growing_cng;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class GrowingCNG_Clustering {

	private static Logger log = Logger.getLogger(GrowingCNG_Clustering.class);

	public static void main(String[] args) {
		final Random r = new Random();
	
		final List<double[]> samples = new ArrayList<double[]>();
		final List<Integer> classMembership = new ArrayList<Integer>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/cng/test2a_nonoise.shp"), true);
		List<String> vars = new ArrayList<String>(sdf.names);
		vars.remove("CLASS");
		
		for (double[] d : sdf.samples) {
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			classMembership.add( (int)d[sdf.names.indexOf("CLASS")] );
			geoms.add(sdf.geoms.get(idx));
		}
				
		int[] fa = new int[vars.size() - 2];
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;
		final int[] ga = new int[] { 0, 1 };
		DataUtils.zScoreColumns(samples, fa);
		
		final Map<Integer,Set<double[]>> m = new HashMap<Integer,Set<double[]>>();
		for( int i = 0; i < samples.size(); i++ ) {
			int cm = classMembership.get(i);
			if( !m.containsKey(cm) )
				m.put(cm, new HashSet<double[]>());
			m.get(cm).add(samples.get(i));
		}
								
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		boolean firstWrite = true;
		for (final double lrB : new double[] { 0.05 }) 
			for (final double lrN : new double[] { lrB/100 })
				for (final int lambda : new int[] { 300 })
					for (final int aMax : new int[] { 100 })
						for( final double alpha : new double[]{ 0.5 } )
							for( final double beta : new double[]{ 0.00005 } )
								//for( double ratio = 0.0; (ratio+=0.01) <= 0.01; )
								for( double ratio = 0.0; (ratio+=0.01) <= 1.0; )
									for( final int distMode : new int[]{ 0, 1, 2, 3, 4, 5 } ){

							final double RATIO = ratio;
							ExecutorService es = Executors.newFixedThreadPool(4);
							List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

							for (int i = 0; i < 32; i++) {
								final int RUN = i;
								futures.add(es.submit(new Callable<double[]>() {

									@Override
									public double[] call() throws Exception {

										List<double[]> neurons = new ArrayList<double[]>();
										for (int i = 0; i < 2; i++) {
											double[] d = samples.get(r.nextInt(samples.size()));
											neurons.add(Arrays.copyOf(d, d.length));
										}

										GrowingCNG ng = new GrowingCNG(neurons, lrB, lrN, gDist, fDist, RATIO, aMax, lambda, alpha, beta );
										ng.samples = samples;
										ng.distMode = distMode;
										ng.run = RUN;
										
										//List<Double> qes = new ArrayList<Double>();
										int t = 1;
										while( true ) { 
											double[] x = samples.get(r.nextInt(samples.size())); 
											ng.train(t++, x);
																															
											/*if (t % 100 == 0)
												qes.add(DataUtils.getMeanQuantizationError(ng.getMapping(samples), fDist));*/
											/*if( t % 100 == 0 )
												qes.add( ng.bErrorSum );*/
											
											if( t == 40000 ) // normal time break
												break;
											/*if( ng.getNeurons().size() == 20 ) // last inserted neuron is not trained at all
												break;*/
										}
																				
										Map<double[], Set<double[]>> mapping = ng.getMapping(samples); 
										List<double[]> usedNeurons = new ArrayList<double[]>(mapping.keySet());
										
										// Connections to connectivity-map
										Map<double[],Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
										for( Connection c : ng.getConections().keySet() ) {
											double[] a = c.getA();
											double[] b = c.getB();
											if( !cm.containsKey(a) )
												cm.put(a, new HashSet<double[]>() );
											if( !cm.containsKey(b) )
												cm.put(b, new HashSet<double[]>() );
											cm.get(a).add(b);
											cm.get(b).add(a);
										}										
										
										// Cluster hierarchical
										List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, fDist, HierarchicalClusteringType.ward);
										List<Set<double[]>> cTree = Clustering.cutTree(tree, 25);
										List<Set<double[]>> cluster = new ArrayList<Set<double[]>>();
										for( Set<double[]> s : cTree ) {
											Set<double[]> sa = new HashSet<double[]>();
											for( double[] d : s )
												if( mapping.containsKey(d))
													sa.addAll(mapping.get(d));
											cluster.add(sa);
										}
																				
										double[] r = new double[] {
												t,
												ng.getNeurons().size(), 
												usedNeurons.size(), 
												ng.getConections().size(), 
												DataUtils.getMeanQuantizationError(mapping, gDist), 
												DataUtils.getMeanQuantizationError(mapping, fDist), 
												ClusterValidation.getNormalizedMutualInformation(m.values(), cluster)
												};
										
										/*double[] r = new double[qes.size()];
										for (int i = 0; i < qes.size(); i++)
											r[i] = qes.get(i);*/
										
										return r;
									}
								}));
							}
							es.shutdown();

							DescriptiveStatistics ds[] = null;
							for (Future<double[]> ff : futures) {
								try {
									double[] ee = ff.get();
									if (ds == null) {
										ds = new DescriptiveStatistics[ee.length];
										for (int i = 0; i < ee.length; i++)
											ds[i] = new DescriptiveStatistics();
									}
									for (int i = 0; i < ee.length; i++)
										ds[i].addValue(ee[i]);
								} catch (InterruptedException ex) {
									ex.printStackTrace();
								} catch (ExecutionException ex) {
									ex.printStackTrace();
								}
							}

							try {
								String fn = "output/result.csv";
								if( firstWrite ) {
									firstWrite = false;
									Files.write(Paths.get(fn), "mode,ratio,t,neurons,used,connections,sqe,fqe,nmi\n".getBytes());
									/*for( int i = 0; i < ds.length; i++ )
										fw.write(",p_"+i);
									fw.write("\n");*/
								}
								String s = "mode_"+distMode+","+ratio;
								for (int i = 0; i < ds.length; i++)
									s += ","+ds[i].getMean();
								s += "\n";
								Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
								System.out.print(s.substring(0, Math.min(s.length(),256)));
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
	}
}

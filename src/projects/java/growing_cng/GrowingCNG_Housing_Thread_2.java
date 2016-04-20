package growing_cng;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
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

import com.vividsolutions.jts.geom.Point;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class GrowingCNG_Housing_Thread_2 {

	private static Logger log = Logger.getLogger(GrowingCNG_Housing_Thread_2.class);

	public static void main(String[] args) {
		final Random r = new Random();
	
		final List<double[]> samples = new ArrayList<double[]>();
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/redcap/Election/election2004.shp"), true);
		
		for (double[] d : sdf.samples) {
			int idx = sdf.samples.indexOf(d);
			Point p = sdf.geoms.get(idx).getCentroid();
			double[] nd = new double[]{p.getX(),p.getY(),d[7]};
			samples.add(nd);
		}
				
		final int[] fa = new int[]{2};
		final int[] ga = new int[] { 0, 1 };
		//DataUtils.zScoreColumns(samples, fa);
										
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		boolean firstWrite = true;
		
		for( final int T_MAX : new int[]{ 100000 } )
		for (final double lrB : new double[] { 0.01 }) // the lower the better for diffError?
			for (final double lrN : new double[] { lrB/100 })
				for (final int lambda : new int[] { 5000 }) // the higher the better for diffError?
					for (final int aMax : new int[] { 200 })
						for( final double alpha : new double[]{ 0.5 } ) 
							for( final double beta : new double[]{ 0.000005 } )
								//for( double ratio : new double[]{ 0.03 } )
								for( double ratio = 0.0; ratio <= 1.0; ratio+=0.01 )
								for( final int distMode : new int[]{ 0, 1, 2, 7, 8 } ){
																														
							final double RATIO = ratio;
							ExecutorService es = Executors.newFixedThreadPool(4);
							List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
							
							for (int i = 0; i < 16; i++) {
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
										
										int t = 1;
										while( true ) { 
											double[] x = samples.get(r.nextInt(samples.size())); 
											ng.train(t, x);
																					
											if( t >= T_MAX ) // time break
												break;
											t++;
										}
										
										Map<double[], Set<double[]>> mapping = ng.getMapping(samples); 
										List<double[]> usedNeurons = new ArrayList<double[]>(mapping.keySet());
																				
										double[] r = new double[] {
												t,
												ng.getNeurons().size(), 
												usedNeurons.size(), 
												ng.getConections().size(), 
												
												DataUtils.getMeanQuantizationError(mapping, gDist),
												DataUtils.getMeanQuantizationError(mapping, fDist), 	
												
												ng.aErrorSum/ng.k,
												ng.bErrorSum/ng.k
												};
																					
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
									Files.write(Paths.get(fn), "lrB,alpha,beta,lambda,amax,mode,ratio,t,neurons,used,connections,finalSqe,finalFqe,aErrorSum,bErrorSum\n".getBytes());
									/*for( int i = 0; i < ds.length; i++ )
										fw.write(",p_"+i);
									fw.write("\n");*/
								}
								String s = lrB+","+alpha+","+beta+","+lambda+","+aMax+","+"mode_"+distMode+","+ratio;
								for (int i = 0; i < ds.length; i++)
									s += ","+ds[i].getMean();
								s += "\n";
								Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
								System.out.println(s.substring(0, Math.min(s.length(),256)));
							} catch (IOException e) {
								e.printStackTrace();
							}
						}
	}
}

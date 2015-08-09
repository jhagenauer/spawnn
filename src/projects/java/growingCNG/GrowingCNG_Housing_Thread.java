package growingCNG;

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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class GrowingCNG_Housing_Thread {

	private static Logger log = Logger.getLogger(GrowingCNG_Housing_Thread.class);

	public static void main(String[] args) {
		final Random r = new Random();
	
		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);
		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");
		vars.add("lnarea_tot");
		vars.add("lnarea_plo");
		vars.add("attic_dum");
		vars.add("cellar_dum");
		vars.add("cond_house_3");
		vars.add("heat_3");
		vars.add("bath_3");
		vars.add("garage_3");
		vars.add("terr_dum");
		vars.add("age_num");
		vars.add("time_index");
		vars.add("zsp_alq_09");
		vars.add("gem_kauf_i");
		vars.add("gem_abi");
		vars.add("gem_alter_");
		vars.add("ln_gem_dic");

		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			geoms.add(sdf.geoms.get(idx));
		}
				
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/cng/test2c.shp"), true);
		List<String> vars = new ArrayList<String>();
		vars.add("X");
		vars.add("Y");
		vars.add("VALUE");
		
		for (double[] d : sdf.samples) {
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			geoms.add(sdf.geoms.get(idx));
		}*/
		
		int[] fa = new int[vars.size() - 2];
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;
		final int[] ga = new int[] { 0, 1 };
		DataUtils.zScoreColumns(samples, fa);
		
		// PCA
		/*int nrComponents = 4;
		List<double[]> ns = DataUtils.removeColumns(samples, ga);
		ns = DataUtils.reduceDimensionByPCA(ns, nrComponents, false);
		for (int k = 0; k < ns.size(); k++) {
			double[] d = ns.get(k);
			double[] nd = new double[ga.length + d.length];
			for (int i = 0; i < ga.length; i++)
				nd[i] = d[ga[i]];
			for (int i = 0; i < nrComponents; i++)
				nd[i + ga.length] = d[i];
			samples.set(k, nd);
		}
		final int[] fFa = new int[nrComponents];
		for (int i = 0; i < nrComponents; i++)
			fFa[i] = i + ga.length;*/
						
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		boolean firstWrite = true;

		/* Was beeinflußt error-calc:
		 * - lrB und lrN, denn langsame adaptation macht initialisierung wichtiger
		 * - lambda, denn jede Einfügung geschieht an error-stelle
		 */
		
		// warum kleiner lernerate auf längere sicht nicht besser? Evtl. weil lrN zu viel disrupted?
			
		for( final int T_MAX : new int[]{80000} )
		for (final double lrB : new double[] { 0.05 }) 
			for (final double lrN : new double[] { lrB/100 })
				for (final int lambda : new int[] { 300 })
					for (final int aMax : new int[] { 100 })
						for( final double alpha : new double[]{ 0.5 } )
							for( final double beta : new double[]{ 0.00005 } )
								for( double ratio = 0.0; (ratio+=0.01) <= 0.01; )
								//for( double ratio = 0.0; (ratio+=0.01) <= 1.0; )
									for( final int distMode : new int[]{ 0 /*, 1, 2, 3, 4, 5*/ } ){

							final double RATIO = ratio;
							ExecutorService es = Executors.newFixedThreadPool(4);
							List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

							for (int i = 0; i < 4; i++) {
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
										
										int t = 0;
										List<Double> qes = new ArrayList<Double>();
										while( t < T_MAX ) { 
											double[] x = samples.get(r.nextInt(samples.size())); 
											ng.train(t++, x); 
											/*if (t % 100 == 0)
												qes.add(DataUtils.getMeanQuantizationError(ng.getMapping(samples), fDist));*/
											/*if( t % 100 == 0 )
												qes.add( ng.bErrorSum );*/
										}
										
										/*double qe = Double.MAX_VALUE;
										while (qe > 3.34) {
											double[] x = samples.get(r.nextInt(samples.size()));
											ng.train(t++, x);
											if (t % 200 == 0) {
												Map<double[], Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), new KangasSorter<double[]>(gDist, fDist, (int) Math.ceil(ng.getNeurons().size() * ratio)));
												qe = DataUtils.getMeanQuantizationError(mapping, fDist);
											}
										}*/
										
										Map<double[], Set<double[]>> mapping = ng.getMapping(samples); 
										int used = 0; 
										for( Set<double[]> s : mapping.values() ) 
											if( !s.isEmpty() ) 
												used++; 
										double[] r = new double[] {
												ng.getNeurons().size(), 
												used, 
												ng.getConections().size(), 
												DataUtils.getMeanQuantizationError(mapping, gDist), 
												DataUtils.getMeanQuantizationError(mapping, fDist), 
												/*ng.rndError, 
												ng.fError*/  
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
									Files.write(Paths.get(fn), "mode,ratio,neurons,used,connections,sqe,fqe\n".getBytes());
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

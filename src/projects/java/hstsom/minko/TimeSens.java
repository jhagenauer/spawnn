package hstsom.minko;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TimeSens {

	private static Logger log = Logger.getLogger(TimeSens.class);

	public static void main(String[] args) {		
		final int T_MAX = 100000;
		
		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> geoDist = new EuclideanDist( new int[] { 0, 1 });
		final Dist<double[]> timeDist = new EuclideanDist( new int[] { 2 });
		final Dist<double[]> fDist = new EuclideanDist( new int[] { 3 });

		final Random rand = new Random();

		final int max_i = 100;
		int threads = 16;

		final Map<double[], Integer> classes = new HashMap<double[], Integer>();
		final List<double[]> samples = new ArrayList<double[]>();
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(args[0]));
			String line = null;
			while ((line = reader.readLine()) != null) {
				String[] s = line.split(",");
				double[] d = { Double.parseDouble(s[0]), Double.parseDouble(s[1]), Double.parseDouble(s[2]), Double.parseDouble(s[3]) };
				samples.add(d);
				classes.put(d, (int) Double.parseDouble(s[4]));
			}
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		class Result {
			public int k, dim_x, dim_y;
			public double nmi, // NMI
					qe, // quant error
					de, // demo error
					te, // time error
					ge, // geo error
					ti; // time
		}

		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<Result>> futures = new ArrayList<Future<Result>>();

		for (final int dim_x : new int[]{ 8, 12, 16, 20 } ) {
			for( final int dim_y : new int[]{1/*, dim_x*/} ) {
				log.debug(dim_x+"x"+dim_y);
				//for (int k = 0; k < (new Grid2DHex<double[]>(dim_x,dim_y)).getMaxDist()/2; k++) {
				for (int k = 0; k < dim_x; k++) {
					final int K = k;
					
					for (int i = 0; i < max_i; i++) {
	
						futures.add(es.submit(new Callable<Result>() {
							@Override
							public Result call() {
	
								Grid2D<double[]> grid = new Grid2DHex<double[]>( dim_x, dim_y );
								SomUtils.initRandom(grid, samples);
								
								BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(timeDist, fDist, K);
	
								SOM som = new SOM(new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5, 0.0), grid, bmuGetter);
								long time = System.currentTimeMillis();
								for (int t = 0; t < T_MAX; t++) {
									double[] x = samples.get(rand.nextInt(samples.size()));
									som.train((double) t / T_MAX, x);
								}
								time = System.currentTimeMillis() - time;
	
								int[][] nImg = SomUtils.getWatershed( 45, 255, 1.0, grid, fDist, false);
								Collection<Set<GridPos>> wsc = SomUtils.getClusterFromWatershed(nImg, grid);
								
								// cluster result
								List<Set<double[]>> clustersA = new ArrayList<Set<double[]>>();
								{
									Map<GridPos, Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
									for (Set<GridPos> c : wsc) {
										Set<double[]> l = new HashSet<double[]>();
										for (GridPos p : c)
											l.addAll(mapping.get(p));
										clustersA.add(l);
									}
								}
	
								// reference clustering
								List<Set<double[]>> clustersB = new ArrayList<Set<double[]>>();
								{
									Map<Integer, Set<double[]>> c = new HashMap<Integer, Set<double[]>>();
									for (double[] d : samples) {
										int cl = classes.get(d);
										if (!c.containsKey(cl)) 
											c.put(cl, new HashSet<double[]>() );
										c.get(cl).add(d);
									}
									
									// merge spatial clusters?
									c.get(1).addAll( c.remove(2) );
									c.get(3).addAll( c.remove(4) );	
																	
									for (Set<double[]> l : c.values())
										clustersB.add(l);
								}
															
								Result r = new Result();
								r.nmi = DataUtils.getNormalizedMutualInformation(clustersA, clustersB);
								if (Double.isNaN(r.nmi)) {
									r.nmi = 0;
									log.debug(dim_x + "x" +dim_y+", r: "+K+" -> NAN");
								}
	
								r.qe = SomUtils.getMeanQuantError(grid, bmuGetter, eDist, samples);
								r.de = SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples);
								r.te = SomUtils.getMeanQuantError(grid, bmuGetter, timeDist, samples);
								r.ge = SomUtils.getMeanQuantError(grid, bmuGetter, geoDist, samples);
								r.ti = time;
								r.k = K;
								r.dim_x = dim_x;
								r.dim_y = dim_y;
								return r;
	
							}
						}));
					}
				}
			}
		}
		es.shutdown();
		
		class ResultContainer {
			public int k, dim_x, dim_y;
			DescriptiveStatistics nmiDs = new DescriptiveStatistics();
		}
		
		List<ResultContainer> results = new ArrayList<ResultContainer>();	
		for (Future<Result> f : futures) {
			try {
				Result r = f.get();
				
				boolean found = false;
				for( ResultContainer rc : results ) {
					if( r.k == rc.k && r.dim_x == rc.dim_x && r.dim_y == rc.dim_y ) {
						rc.nmiDs.addValue(r.nmi);
						found = true;
						break;
					}
				}
				
				if( !found ) {
					ResultContainer rc = new ResultContainer();
					rc.k = r.k;
					rc.dim_x = r.dim_x;
					rc.dim_y = r.dim_y;
					rc.nmiDs.addValue( r.nmi );
					results.add(rc);
				}
					
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		
		FileWriter fw = null;
		try {
			fw = new FileWriter("time_"+args[0].split("/")[2]);
		
			// output
			fw.write("radius,x,y,min,max,mean,stdev\n");
			for( ResultContainer rc : results ) 
				fw.write(rc.k+", "+rc.dim_x+","+rc.dim_y+","+rc.nmiDs.getMin()+","+rc.nmiDs.getMax()+","+rc.nmiDs.getMean()+","+rc.nmiDs.getStandardDeviation()+"\n" );
			
			fw.close();
		} catch( Exception e ) {
			e.printStackTrace();
		}
	}
}

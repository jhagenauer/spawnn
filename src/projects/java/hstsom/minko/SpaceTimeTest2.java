package hstsom.minko;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SampleBuilder;

public class SpaceTimeTest2 {
		
	private static Logger log = Logger.getLogger(SpaceTimeTest2.class);
	
	public static void main(String[] args) {
		int T_MAX = 100000;

		int GEO_DIM_X = 16;
		int GEO_DIM_Y = 16;
		/* grüne in cluster -> map zu groß, radius zu niedrig, abst. zu groß
		 * cluster zusammen -> map zu klein, radius zu hoch, abstand zu klein
		 */
		int GEO_RADIUS = 4;
		
		// 1d -> 30,1,10; 2d -> 
		int TIME_DIM_X = 16;
		int TIME_DIM_Y = 1;
		int TIME_RADIUS = 4;
				
		/* Prolbem: Radius zu niedrig-> geocluster wird gespalten
		 * 			Radius zu hoch   -> geocluster werden nicht differenztiert
		 */
		
		int[] ga = new int[]{0,1};
									
		Dist<double[]> eDist = new EuclideanDist();
		Dist<double[]> geoDist = new EuclideanDist( ga );
		Dist<double[]> timeDist = new EuclideanDist( new int[]{2} );
		Dist<double[]> fDist = new EuclideanDist( new int[]{3} );
		
		Random r = new Random();
				
		final Map<double[],Integer> classes = new HashMap<double[],Integer>();
		final List<double[]> samples = new ArrayList<double[]>();
		
		BufferedReader reader = null;
		try {
			SampleBuilder.buildSPSDataSet(100000, 0.04, 0.01, "data/tmp.csv" );
			reader = new BufferedReader( new FileReader("data/tmp.csv") );
			String line = null;
			while( ( line = reader.readLine() ) != null ) {
				String[] s = line.split(",");
				double[] d = { Double.parseDouble(s[0]), Double.parseDouble(s[1]), Double.parseDouble(s[2]), Double.parseDouble(s[3])};
				samples.add(d);
				classes.put( d, (int)Double.parseDouble(s[4]) );
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
				
		// geo som
		Grid2D<double[]> gGrid;
		BmuGetter<double[]> gBg;
		{
			Grid2D<double[]> grid = new Grid2DHex<double[]>(GEO_DIM_X, GEO_DIM_Y );
			SomUtils.initRandom(grid, samples);
									
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>( geoDist, fDist, GEO_RADIUS );
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
			long time = System.currentTimeMillis();
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x );
			}
			time = System.currentTimeMillis() - time;
						
			log.debug(" --- GeoSom  --- ");
			/*log.debug("quantError: "+SomUtils.getQuantError( grid, bmuGetter, fDist, samples ) );
			log.debug("geoError: " +SomUtils.getQuantError( grid, bmuGetter, geoDist, samples ) );
			log.debug("topoError: "+SomUtils.getTopoError( grid, bmuGetter, samples ) );*/
			log.info("Took: "+time+"ms");
														
			try {
				Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter );
				SomUtils.printUMatrix( grid, fDist, new FileOutputStream( "output/gsomUmat.png" ) );
				//SomUtils.printUMatrix( grid, geoDist, new FileOutputStream( "output/gsomUmatGeo.png" ) );
				SomUtils.printClassDist( classes, bmus, grid, new FileOutputStream("output/gsomClass.png") );	
				SomUtils.printGeoGrid( ga, grid, new FileOutputStream("output/gtopo.png") );
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} 
			
			int[][] nImg = SomUtils.getWatershed( 45, 255, 1.0, grid, fDist, false);
			Collection<Set<GridPos>> wsc = SomUtils.getClusterFromWatershed(nImg, grid);
			List<Set<double[]>> clustersA = new ArrayList<Set<double[]>>();
			{
				Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
				for (Set<GridPos> c : wsc) {
					Set<double[]> l = new HashSet<double[]>();
					for (GridPos p : c)
						l.addAll(mapping.get(p));
					clustersA.add(l);
				}
			}

			List<Set<double[]>> clustersB = new ArrayList<Set<double[]>>();
			{
				Map<Integer, Set<double[]>> c = new HashMap<Integer, Set<double[]>>();
				for (double[] d : samples) {
					int cl = classes.get(d);
					if (c.containsKey(cl))
						c.get(cl).add(d);
					else {
						Set<double[]> l = new HashSet<double[]>();
						l.add(d);
						c.put(cl, l);
					}
				}
				
				// merge temporal clusters
				c.get(1).addAll( c.remove(3) );
				c.get(2).addAll( c.remove(4) );	
				
				for (Set<double[]> l : c.values())
					clustersB.add(l);
			}
			
			log.debug("NMI: "+DataUtils.getNormalizedMutualInformation(clustersA, clustersB));
			
			gGrid = grid;
			gBg = bmuGetter;
		}
								
		// temp som
		Grid2D<double[]> tGrid;
		BmuGetter<double[]> tBg;
		{
			Grid2D<double[]> grid = new Grid2DHex<double[]>(TIME_DIM_X, TIME_DIM_Y );
			SomUtils.initLinear(grid, samples, true);
						
			// TODO: Wichtig: Zeit ist unidirektional, dass wird hier nicht berücksichtigt
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>( timeDist, fDist, TIME_RADIUS );
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
			long time = System.currentTimeMillis();
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x );
			}
			time = System.currentTimeMillis() - time;
			
			log.debug(" --- TimeSom --- ");
			/*log.debug("quantError: "+SomUtils.getQuantError( grid, bmuGetter, fDist, samples ) );
			log.debug("timeError: " +SomUtils.getQuantError( grid, bmuGetter, timeDist, samples ) );
			log.debug("topoError: "+SomUtils.getTopoError( grid, bmuGetter, samples ) );*/
			log.info("Took: "+time+"ms");
			
			try {
				Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter );
				SomUtils.printUMatrix( grid, fDist, new FileOutputStream( "output/tsomUmat.png" ) );
				SomUtils.printClassDist( classes, bmus, grid, new FileOutputStream("output/tsomClass.png") );				
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} 
			
			int[][] nImg = SomUtils.getWatershed( 45, 255, 1.0, grid, fDist, false);
			Collection<Set<GridPos>> wsc = SomUtils.getClusterFromWatershed(nImg, grid);
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

			List<Set<double[]>> clustersB = new ArrayList<Set<double[]>>();
			{
				Map<Integer, Set<double[]>> c = new HashMap<Integer, Set<double[]>>();
				for (double[] d : samples) {
					int cl = classes.get(d);
					if (c.containsKey(cl))
						c.get(cl).add(d);
					else {
						Set<double[]> l = new HashSet<double[]>();
						l.add(d);
						c.put(cl, l);
					}
				}
				
				c.get(1).addAll( c.remove(2) );
				c.get(3).addAll( c.remove(4) );	
				
				for (Set<double[]> l : c.values())
					clustersB.add(l);
			}

			log.debug("NMI: "+DataUtils.getNormalizedMutualInformation(clustersA, clustersB));
								
			tGrid = grid;
			tBg = bmuGetter;
		}
		
		List<double[]> l = new ArrayList<double[]>();
		for( double[] x : samples ) {										
			int[] a = gBg.getBmuPos(x, gGrid ).getPosVector();
			int[] b = tBg.getBmuPos(x, tGrid ).getPosVector();
															
			l.add( new double[]{ a[0], a[1], b[0] } );
		}
								
		DataUtils.normalize(l);
		//DataUtil.zScore(l);
										
		// rebuild classes
		for( int i = 0; i < samples.size(); i++ ) {
			double[] d = samples.get(i);
			double[] nd = l.get(i);
			
			int c = classes.get(d);
			classes.remove(d);
			classes.put( nd, c );
		}
		
		// h som
		{	
			Grid2D<double[]> grid = new Grid2DHex<double[]>( 8, 8 );
			SomUtils.initRandom(grid, l);
								
			//BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>( new SGridDist<double[]>(gGrid, tGrid) );
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>( eDist );
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist() ), new LinearDecay(0.5,0.0), grid, bmuGetter );
			long time = System.currentTimeMillis();
			for (int t = 0; t < T_MAX; t++) {
				double[] x = l.get(r.nextInt(l.size() ) );								       							
				som.train( (double)t/T_MAX, x );
			}
			time = System.currentTimeMillis() - time;
																	
			log.debug(" --- HierarchcialSom --- ");		
			/*log.debug("quantError: "+SomUtils.getQuantError( grid, bmuGetter, eDist, l ) );
			log.debug("topoError: "+SomUtils.getTopoError( grid, bmuGetter, l ) );	
			log.info("Took: "+time+"ms");*/
												
			try {
				Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(l, grid, bmuGetter );
				//SomUtils.printDMatrix( grid, eDist, new FileOutputStream( "output/hsomDmat.png" ) );
				SomUtils.printUMatrix( grid, eDist, new FileOutputStream( "output/hsomUmat.png" ) );
				SomUtils.printClassDist( classes, bmus, grid, new FileOutputStream("output/hsomClass.png") );
				
				int[][] nImg = SomUtils.getWatershed( 45, 255, 1.0, grid, eDist, false); 
				Collection<Set<GridPos>> wsc = SomUtils.getClusterFromWatershed(nImg, grid); 
				SomUtils.printClusters( wsc, grid, new FileOutputStream( "output/hsomCluster.png" ) );
				
				List<Set<double[]>> clustersA = new ArrayList<Set<double[]>>();
				{
					for( Set<GridPos> c : wsc ) {
						Set<double[]> li = new HashSet<double[]>();
						for( GridPos p : c ) 
							li.addAll( bmus.get(p) );
						
						/*HashSet<double[]> h = new HashSet<double[]>(li);
						li.clear();
						li.addAll(h);*/
						   
						clustersA.add(li);
					}
				}
												
				List<Set<double[]>> clustersB = new ArrayList<Set<double[]>>();
				{
					Map<Integer,Set<double[]>> c = new HashMap<Integer,Set<double[]>>();
					for( double[] d : l ) {
						int cl = classes.get(d);
						if( c.containsKey(cl))
							c.get(cl).add(d);
						else {
							Set<double[]> li = new HashSet<double[]>();
							li.add(d);
							c.put(cl,li);
						}
					}
					for( Set<double[]> li: c.values() ) {
						
						/*HashSet<double[]> h = new HashSet<double[]>(li);
					    li.clear();
					    li.addAll(h);*/
					
						clustersB.add(li);
					}
				}		
				
				log.info( "NMI: "+DataUtils.getNormalizedMutualInformation( clustersA, clustersB ) );
								
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
}

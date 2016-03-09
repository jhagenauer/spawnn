package hstsom.austria;


import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.geotools.data.DataStore;
import org.geotools.data.FeatureStore;
import org.geotools.data.FileDataStoreFactorySpi;
import org.geotools.data.shapefile.ShapefileDataStoreFactory;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.FeatureCollection;
import org.geotools.feature.FeatureCollections;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.DefaultMapContext;
import org.geotools.map.FeatureLayer;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;

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

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.MultiPolygon;

public class AustriaMunicipalities {

	private static Logger log = Logger.getLogger(AustriaMunicipalities.class);
		
	public static void main(String[] args) {
		
		/* 2d-space: 10,8,2 
		 * 1d-space: 12,1,2
		 * 
		 * 2d-time: ~ 10,8,3 
		 * 1d-time: 12,1,3
		 */

		Random r = new Random();
				
		// 2d space
		// 14x12, normalisiert: 3, zScore: 4
		int geo_x = 14;
		int geo_y = 12;
		int geo_k = 4;
		
		// 1d time
		// 12x1, normalisiert: 3, zScore: 5
		int time_x = 12;
		int time_y = 1;
		int time_k = 5;
		
		int X_SOM = 3;
		int Y_SOM = 3;
		int T_MAX = 1000000;
		
		File shape = new File("data/sps/munaus_3.shp");
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(shape, new int[]{}, true);
		final List<double[]> origSamples = DataUtils.readSamplesFromShapeFile(shape, new int[]{}, false);
		final List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(shape);
		
		//String[] names = new String[]{"YEAR","X","Y","AREA","BEV","BEVDENS","LNBVDENS","BES","BESRATE","AUSP","AUSPRATE","ALLE","ALLEDENS","LNALLDNS"};
		String[] names = new String[]{"YEAR","X","Y","AREA","BEV","BEVDENS","POP","BES","EE","AUSP","COM","ALLE","ALLEDENS","NAWP"};
			
		int[] ta = new int[]{0}; 
		int[] ga = new int[]{1,2}; 
		int[] fa = new int[]{6,8,10,13}; 				
		
		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> fDist = new EuclideanDist( fa);
		final Dist<double[]> gDist = new EuclideanDist( ga);
		final Dist<double[]> tDist = new EuclideanDist( ta);
		
		DataUtils.normalizeColumns(samples, ta);
		DataUtils.normalizeGeoColumns(samples, ga);
		DataUtils.normalizeColumns(samples, fa);
		
		/*DataUtils.zScoreColumns(samples, ta);
		DataUtils.zScoreGeoColumns(samples,ga,gDist);
		DataUtils.zScoreColumns(samples, fa);*/
		
		// geosom som
		for( int k = 0; k <= 3; k++ ){
			log.debug(" --- Kangas Som, radius "+k+"  --- ");
			
			Grid2D<double[]> grid = new Grid2DHex<double[]>(X_SOM, Y_SOM );
			SomUtils.initRandom(grid, samples );
						
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>( gDist, fDist, k);
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x );
			}
						
			double dqe = SomUtils.getMeanQuantError( grid, bmuGetter, fDist, samples );
			double sqe = SomUtils.getMeanQuantError( grid, bmuGetter, gDist, samples );
			double tqe = SomUtils.getMeanQuantError( grid, bmuGetter, tDist, samples );
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			ds.addValue(dqe);
			ds.addValue(sqe);
			ds.addValue(tqe);
			
			log.debug("dqe: "+dqe);
			log.debug("sqe: "+sqe);
			log.debug("tqe: "+tqe);
			log.debug("mean: "+ds.getMean());
			log.debug("sd: "+ds.getStandardDeviation());
		}				
			
				
		// geo som
		Grid2D<double[]> gGrid, tGrid;
		BmuGetter<double[]> gBg, tBg;
		{
			Grid2D<double[]> grid = new Grid2DHex<double[]>(geo_x, geo_y );
			//SomUtils.initLinear(grid, samples, true);
			SomUtils.initRandom(grid, samples);
						
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>( gDist, fDist, geo_k );
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x );
			}
			
			log.debug(" --- GeoSom  --- ");
			log.debug("quantError: "+SomUtils.getMeanQuantError( grid, bmuGetter, fDist, samples ) );
			log.debug("geoError: " +SomUtils.getMeanQuantError( grid, bmuGetter, gDist, samples ) );
			log.debug("topoError: "+SomUtils.getTopoError( grid, bmuGetter, samples ) );
							
			try {
				SomUtils.printUMatrix( grid, fDist, new FileOutputStream( "output/gsomUmat.png" ) );
				SomUtils.printGeoGrid(ga, grid, new FileOutputStream("output/gTopo.png") );
				SomUtils.saveGrid( grid, new FileOutputStream("output/gsom.xml") );
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} 
			
			gGrid = grid;
			gBg = bmuGetter;
		}
				
		// temp som
		{
			Grid2D<double[]> grid = new Grid2DHex<double[]>(time_x, time_y );
			//SomUtils.initLinear(grid, samples, true);
			SomUtils.initRandom(grid, samples);
			
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>( tDist, fDist, time_k );
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x );
			}
			
			log.debug(" --- TimeSom --- ");
			log.debug("quantError: "+SomUtils.getMeanQuantError( grid, bmuGetter, fDist, samples ) );
			log.debug("geoError: " +SomUtils.getMeanQuantError( grid, bmuGetter, gDist, samples ) );
			log.debug("topoError: "+SomUtils.getTopoError( grid, bmuGetter, samples ) );
									
			try {
				SomUtils.printUMatrix( grid, fDist, new FileOutputStream( "output/tsomUmat.png" ) );
				SomUtils.saveGrid( grid, new FileOutputStream("output/tsom.xml") );
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} 		
			tGrid = grid;
			tBg = bmuGetter;
		}
		
		// h som
		{	
			log.debug(" --- HierarchcialSom --- ");		
			
			List<double[]> l = new ArrayList<double[]>();
			for( double[] x : samples ) {										
				int[] geoPos = gBg.getBmuPos(x, gGrid ).getPosVector();
				int[] timePos = tBg.getBmuPos(x, tGrid ).getPosVector();
														
				l.add( new double[]{ geoPos[0],geoPos[1],timePos[0]/*,timePos[1]*/ } );
			}
			log.debug("HSTSOM input vector size: "+l.get(0).length);
												
			DataUtils.normalize(l);
			// DataUtil.zScore(l);
									
			Grid2D<double[]> grid = new Grid2DHex<double[]>( X_SOM, Y_SOM );
			SomUtils.initRandom(grid, l);
			//grid.initLinear(l, true);
						
			//NOTE: bei sGrid nicht normalisieren und komplette vektoren in l speichern
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>( eDist );
			//BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>( new SGridDist<double[]>(gGrid,tGrid) );
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
			for (int t = 0; t < T_MAX; t++) {
				double[] x = l.get(r.nextInt(l.size() ) );	
				som.train( (double)t/T_MAX, x );
			}
						
			double dqe = SomUtils.getMeanQuantError( grid, bmuGetter, fDist, samples );
			double sqe = SomUtils.getMeanQuantError( grid, bmuGetter, gDist, samples );
			double tqe = SomUtils.getMeanQuantError( grid, bmuGetter, tDist, samples );
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			ds.addValue(dqe);
			ds.addValue(sqe);
			ds.addValue(tqe);
			
			log.debug("dqe: "+dqe);
			log.debug("sqe: "+sqe);
			log.debug("tqe: "+tqe);
			log.debug("mean: "+ds.getMean());
			log.debug("sd: "+ds.getStandardDeviation());		
												
			try {
				SomUtils.printUMatrix( grid, eDist, new FileOutputStream( "output/hsomUmat.png" ) );
				SomUtils.saveGrid( grid, new FileOutputStream("output/hsom.xml") );
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
						        	
        	Map<GridPos,Set<double[]>> bmuMap = SomUtils.getBmuMapping(l, grid, bmuGetter);
        	Map<GridPos,Set<double[]>> mapping = new HashMap<GridPos,Set<double[]>>();
        	
        	for( GridPos p : bmuMap.keySet() ) {
        		Set<double[]> orig = new HashSet<double[]>();
				for( double[] d : bmuMap.get(p) ) 
					orig.add( samples.get(l.indexOf(d)) );
				mapping.put( p, orig);
        	}
        	
        	// print sse
        	Map<double[],Set<double[]>> sseMap = new HashMap<double[],Set<double[]>>();
	        for( Set<double[]> s :mapping.values() )
	        	sseMap.put( DataUtils.getMean(s), s);
	        
	        double qed = DataUtils.getMeanQuantizationError(sseMap, fDist);
	        double qeg = DataUtils.getMeanQuantizationError(sseMap, gDist);
	        double qet = DataUtils.getMeanQuantizationError(sseMap, tDist);
	        	 
	        log.debug("QE Demo: "+qed );
	        log.debug("QE Geo: "+ qeg );
	        log.debug("QE Temp: "+ qet );
	        log.debug("sum: "+(qed+qeg+qet));
        	        	
        	List<GridPos> sortedList = new ArrayList<GridPos>(grid.getPositions());
        	Collections.sort(sortedList);
        	
        	// safe shape-file with clusters
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("cluster");
			typeBuilder.add("CLUSTER",Integer.class);
			typeBuilder.add("the_geom",MultiPolygon.class);
			
			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder( typeBuilder.buildFeatureType() );
			DefaultFeatureCollection fcCluster = new DefaultFeatureCollection();
        	
        	// save attribute-value-mapping and build shape file
        	FileWriter fw;
        	FileWriter fw2;
			try {
				fw = new FileWriter(new File("output/cluster_kv.csv") ); // stores normalized values
				fw.write("attribute,value,cluster\n");
				
				fw2 = new FileWriter(new File("output/cluster_orig.csv") ); // stores orig values
				for( String n : names )
					fw2.write(n+",");
				fw2.write("cluster\n");
				
	        	int k = 0;
	        	for( GridPos p : sortedList ) {
	        		for( double[] d : mapping.get(p) ) {
	        			
	        			int idx = samples.indexOf(d);
	        			double[] origD = origSamples.get(idx);
	        			Geometry geom = geoms.get(idx);
	        			
	        			// shape
	        			featureBuilder.set("CLUSTER",k);
	        			featureBuilder.set("the_geom",geom);
	        			fcCluster.add( featureBuilder.buildFeature(""+fcCluster.size()));
	        				    
	        			// attribute-value
	        			for( int i = 0; i < names.length; i++ )
	        				fw.write(names[i]+","+d[i]+",Cluster "+k+"\n");
	        			
	        			// orig-csv
	        			for( int i = 0; i < names.length; i++ )
	        				fw2.write(origD[i]+",");
	        			fw2.write(k+"\n");
	        			
	        		}
	        		k++;
	        	}
	        	
	        	fw.close();
	        	fw2.close();
			} catch (IOException e1) {
				e1.printStackTrace();
        	}
			
			// store
			//write shape 
			try {
				File out = new File("output/cluster.shp");
				Map map = Collections.singletonMap("url", out.toURI().toURL());
		        FileDataStoreFactorySpi factory = new ShapefileDataStoreFactory();
				DataStore myData = factory.createNewDataStore(map);
				myData.createSchema((SimpleFeatureType) fcCluster.getSchema());
				String n = fcCluster.getSchema().getName().getLocalPart();
				FeatureStore<SimpleFeatureType, SimpleFeature> store = (FeatureStore<SimpleFeatureType, SimpleFeature>) myData.getFeatureSource(n);
				store.addFeatures(fcCluster);
			} catch (MalformedURLException ex) {
				ex.printStackTrace();
			} catch (IOException ex) {
				ex.printStackTrace();
			}
			
			//log.debug(fcCluster.getBounds().getWidth()/fcCluster.getBounds().getHeight());
			
			// write image 
			{
				Map<GridPos,BufferedImage> imgMap = new HashMap<GridPos,BufferedImage>();
				for( GridPos p : sortedList ) {
					//log.debug(p);
					Set<double[]> cluster = mapping.get(p);
									
					// build cluster
					DefaultFeatureCollection fc = new DefaultFeatureCollection();
					
					for( double[] d : cluster ) {
						int idx = samples.indexOf(d);
						featureBuilder.set("CLUSTER",sortedList.indexOf(p));
						featureBuilder.set("the_geom", geoms.get( idx ) );
						fc.add( featureBuilder.buildFeature(""+fc.size()) );
					}
									
					if( cluster.size() == 0 ) {
						log.warn("Empty cluster!");
						continue;
					}
					
					int[] decades = new int[5];
				    for( int i = 0; i < decades.length; i++ )
				     	decades[i] = 0;
				    
				    // get min and max time
				    double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
				    for( double[] s : samples )
				     	if( s[ta[0]] < min )
				       		min = s[ta[0]];
				       	else if( s[ta[0]] > max )
				       		max = s[ta[0]];
				    double dur = (max-min)/(decades.length-1);
				        	        
				    for( double[] s : cluster ) {
				      	int h = (int)((s[ta[0]]-min)/dur);
				       	decades[h]++;
				    }
				     			    
				    int maxDec = 0;
					for( int d : decades )
						if( d > maxDec )
							maxDec = d;
																								
					StyleBuilder sb = new StyleBuilder();
					DefaultMapContext map = new DefaultMapContext();
			        try {
			        	map.addLayer( new FeatureLayer( fcCluster, SLD.wrapSymbolizers( sb.createPolygonSymbolizer( Color.BLACK, 1.0 ) ) ) );
			        	map.addLayer( new FeatureLayer( fc, SLD.wrapSymbolizers( sb.createPolygonSymbolizer( Color.BLUE, Color.BLACK, 1.0 ) ) ) );
			        	map.addLayer( new FeatureLayer( fc, SLD.wrapSymbolizers( sb.createTextSymbolizer(Color.RED, sb.createFont("Arial",24), "CLUSTER"))) );
					} catch (Exception e1) {
						e1.printStackTrace();
					}
							       
			        GTRenderer renderer = new StreamingRenderer();
			        renderer.setContext( map );
			        
			        Rectangle imageBounds=null;
			        try{
			            ReferencedEnvelope mapBounds=map.getLayerBounds();
			            double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			            int imageWidth = 1000;
			            imageBounds = new Rectangle( 0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
			       	        
			            BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB); 
			       	  	Graphics2D gr = image.createGraphics();
					    
			       	  	gr.setPaint(Color.WHITE);
					    gr.fill(imageBounds);
			            
					    renderer.paint(gr, imageBounds, map.getAreaOfInterest());
			            
			            // paint temp clusters into picture
			            int yOffset = 50;
			        	for( int i = 0; i < decades.length; i++ ) {
							
							gr.setColor( Color.BLUE );
							
							int v = 100 * decades[i]/maxDec; // höhe
							gr.fillRect(i*100, imageBounds.height-v-yOffset, 100, v );
							 
							gr.setColor(Color.BLACK);
							gr.drawRect(i*100, imageBounds.height-100-yOffset, 99, 99);
						}
			        	image.flush();
			        	imgMap.put(p,image);
			              
			        } catch(IOException e){
			            e.printStackTrace();
			        }
			        map.dispose();    
				}
				
				// output imageMap
				int width = imgMap.values().iterator().next().getWidth()+20;
				int height = imgMap.values().iterator().next().getHeight()+20;
				
				BufferedImage image = new BufferedImage(grid.getSizeOfDim(0)*width, grid.getSizeOfDim(1)*height, BufferedImage.TYPE_INT_ARGB);
				Graphics2D g = image.createGraphics();
				
				for( GridPos p : imgMap.keySet() ) 
					g.drawImage(imgMap.get(p), p.getPos(0)*width, p.getPos(1)*height, null);
				
				try {
					ImageIO.write( image, "png", new FileOutputStream("output/hstsom_austria"+3+"x"+3+".png"));
				} catch (IOException ex) {
					ex.printStackTrace();
				}
			}
			
			System.out.println("done");
		}
	}
}

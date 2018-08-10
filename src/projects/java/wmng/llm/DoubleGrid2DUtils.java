package wmng.llm;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.Grid2DToroid;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class DoubleGrid2DUtils {
	
	private static Logger log = Logger.getLogger(DoubleGrid2DUtils.class);
		
	public static Grid2D_Map<double[]> createSpDepGrid(int xDim, int yDim, boolean toroid ) {
		
		Random r = new Random();
		Grid2D_Map<double[]> grid;
		if( toroid )
			grid = new Grid2DToroid<double[]>(xDim, yDim);
		else
			grid = new Grid2D_Map<double[]>(xDim, yDim);
		
			List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
			Collections.shuffle(pos);
			
		Map<GridPos,Double> n = new HashMap<GridPos,Double>();
		for( GridPos p : pos ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			grid.setPrototypeAt(p, new double[]{ i, j, 0, 0 });
			n.put(p, r.nextDouble() );
		}
			
		/*for( GridPos p : pos ) {
			DescriptiveStatistics dsN = new DescriptiveStatistics();
			dsN.addValue(n.get(p));
			for(GridPos nb : grid.getNeighbours(p) )
				dsN.addValue( n.get(nb) );
			grid.getPrototypeAt(p)[2] = dsN.getMean();
		}
		for( GridPos p : pos ) {
			DescriptiveStatistics dsA = new DescriptiveStatistics();
			for( GridPos nb : grid.getNeighbours(p) )
				dsA.addValue( grid.getPrototypeAt(nb)[2] );
			grid.getPrototypeAt(p)[3] = grid.getPrototypeAt(p)[2] + dsA.getMean() + 0.5 * r.nextDouble();
		}*/
			
		// works
		/*for( GridPos p : pos ) 
			grid.getPrototypeAt(p)[2] = n.get(p);
		
		for( GridPos p : pos ) {
			DescriptiveStatistics dsA = new DescriptiveStatistics();
			for( GridPos nb : grid.getNeighbours(p) ) {
				DescriptiveStatistics dsB = new DescriptiveStatistics();
				
				for( GridPos nb2 : grid.getNeighbours(nb) )
					if( nb2 != p )
						dsB.addValue( grid.getPrototypeAt(nb2)[2] );
				dsA.addValue( Math.pow( grid.getPrototypeAt(nb)[2] * dsB.getMean(), 2) );
			}
			grid.getPrototypeAt(p)[3] = Math.sqrt( Math.pow(grid.getPrototypeAt(p)[2],2) + dsA.getMean());
			// grid.getPrototypeAt(p)[3] = grid.getPrototypeAt(p)[2] + dsA.getMean();
		}*/
		
		/*for( GridPos p : pos ) {
			DescriptiveStatistics dsA = new DescriptiveStatistics();
			DescriptiveStatistics dsB = new DescriptiveStatistics();
			
			for( GridPos nb : grid.getNeighbours(p) ) {
				dsA.addValue( grid.getPrototypeAt(nb)[2] );
				for( GridPos nb2 : grid.getNeighbours(nb) )
					if( nb2 != p )
						dsB.addValue( grid.getPrototypeAt(nb2)[2] );
			}
			grid.getPrototypeAt(p)[3] = Math.sqrt(Math.pow(grid.getPrototypeAt(p)[2],2) + Math.pow(dsA.getMean(),2) * Math.pow(dsB.getMean(),2) );
			//grid.getPrototypeAt(p)[3] = grid.getPrototypeAt(p)[2] + dsA.getMean();
		}*/
		
		for( GridPos p : pos ) 
			grid.getPrototypeAt(p)[2] = n.get(p);
		for( int k = 0; k < 2500; k++ ) {
			GridPos p = pos.get(r.nextInt(pos.size()));
			DescriptiveStatistics ds = new DescriptiveStatistics();
			ds.addValue( grid.getPrototypeAt(p)[2] );
			for( GridPos nb : grid.getNeighbours(p) )
				ds.addValue( grid.getPrototypeAt(nb)[2] );
			//grid.getPrototypeAt(p)[3] = grid.getPrototypeAt(p)[2];
			grid.getPrototypeAt(p)[2] = ds.getMean();
		}
		for( GridPos p : pos ) 
			grid.getPrototypeAt(p)[3] = n.get(p);
			
		return grid;
	}

	public static List<Geometry> getGeoms(List<GridPos> positions ) {
		GeometryFactory gf = new GeometryFactory();
		List<Geometry> geoms = new ArrayList<Geometry>();
		for( GridPos p : positions ) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			geoms.add( gf.createPolygon(new Coordinate[]{
					new Coordinate(i*50, j*50),
					new Coordinate((i+1)*50, j*50),
					new Coordinate((i+1)*50, (j+1)*50),
					new Coordinate(i*50, (j+1)*50),
					new Coordinate(i*50, j*50),
			}) );
		}
		return geoms;
	}
	
	public static SpatialDataFrame gridToSDF(Grid2D_Map<double[]> grid) {
		GeometryFactory gf = new GeometryFactory();
		List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		for( GridPos p : pos ) {
			samples.add( grid.getPrototypeAt(p) );
			
			int x = p.getPos(0);
			int y = p.getPos(1);
			int size = 50;
			geoms.add( gf.createPolygon( new Coordinate[]{
				new Coordinate(x*size,y*size),
				new Coordinate(x*size,(y+1)*size),
				new Coordinate((x+1)*size,(y+1)*size),
				new Coordinate((x+1)*size,y*size),
				new Coordinate(x*size,y*size),
			}));
		}
		SpatialDataFrame sdf = new SpatialDataFrame();
		sdf.geoms = geoms;
		sdf.samples = samples;
		sdf.names = new ArrayList<String>();
		sdf.names.add("xPos");
		sdf.names.add("yPos");
		sdf.names.add("x1");
		sdf.names.add("y");
		return sdf;
	}
		
	public static void main(String[] args) {
		DecimalFormat df = new DecimalFormat("0000");
		
		for( int i = 0; i < 1; i++ ){
			log.debug(i);
			Grid2D_Map<double[]> grid = createSpDepGrid(50,50,true);
						
			try {
				GZIPOutputStream gzos = new GZIPOutputStream(new FileOutputStream("output/grid_"+df.format(i)+".xml.gz"));
				SomUtils.saveGrid(grid,  gzos );
				gzos.finish();
				gzos.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			if( i == 0 ) {
				SpatialDataFrame sdf = gridToSDF(grid);
				DataUtils.writeShape(sdf.samples, sdf.geoms, sdf.getNames(), "output/grid.shp");
				Drawer.geoDrawValues(sdf, 2, ColorBrewer.Blues, "output/x1.png");
				Drawer.geoDrawValues(sdf, 3, ColorBrewer.Blues, "output/y.png");
				
				Map<double[],Map<double[],Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeightsOld(GeoUtils.getNeighborsFromGrid(grid)));
				for( int j = 2; j <= 3; j++ ) {
					
					Map<double[],Double> values = new HashMap<double[],Double>();
					for( double[] d : sdf.samples )
						values.put(d,d[j]);
					double[] moran = GeoUtils.getMoransIStatisticsMonteCarlo(dMap, values, 999);
					log.debug("var "+j+", moran: "+moran[0]+", p-Value: "+moran[4]);
				}
			}
		}
	}
}

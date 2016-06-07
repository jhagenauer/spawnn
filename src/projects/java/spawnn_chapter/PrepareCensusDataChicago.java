package spawnn_chapter;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.geotools.referencing.CRS;
import org.opengis.geometry.MismatchedDimensionException;
import org.opengis.referencing.FactoryException;
import org.opengis.referencing.NoSuchAuthorityCodeException;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphUtils;
import spawnn.utils.SpatialDataFrame;

public class PrepareCensusDataChicago {

	public static void main(String[] args) {

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/chicago/dec10_sf1dp1/DEC_10_SF1_SF1DP1_with_ann.csv"), new int[]{}, false);
		for( int i = 0; i < df.names.size(); i++ )
			System.out.println(i+","+df.names.get(i)+","+i);
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/chicago/boundaries_tracts/Export_Output.shp"), new int[]{1,2,3}, true);
				
		Map<double[], Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(sdf.samples, sdf.geoms,false);
		List<Map<double[], Set<double[]>>> sub = GraphUtils.getSubGraphs(cm);

		/*Map<double[], Set<double[]>> largestSub = null;
		for (Map<double[], Set<double[]>> s : sub) {
			if (largestSub == null || largestSub.size() < s.size())
				largestSub = s;
		}
		Set<double[]> samples = GraphUtils.getNodes(largestSub);*/

		try {
			CoordinateReferenceSystem targetCRS = sdf.crs; //CRS.decode("EPSG:102008");
			MathTransform transform = CRS.findMathTransform(sdf.crs, targetCRS, true); // requires epsg-extensions-plugin
			List<double[]> nSamples = new ArrayList<double[]>();
			List<Geometry> nGeoms = new ArrayList<Geometry>();
			for (double[] n : sdf.samples ) {
				int state = (int) n[0];
				int county = (int) n[1];
				int tract = (int)(n[2]);

				for (double[] d : df.samples) {
					String id = (long) d[0] + "";
					
					int s = Integer.parseInt(id.substring(0, 2)); // first digits, state			
					int c = Integer.parseInt(id.substring(2, 5)); // last 3 digits, county
					int t = Integer.parseInt(id.substring(5,id.length()));

					if ( t != tract )
						continue;
					
					int idx = sdf.samples.indexOf(n);
					Geometry g = sdf.geoms.get(idx); // JTS.transform(sdf.geoms.get(idx), transform);
					double totPop = d[25]+d[50];
					double[] nd = new double[] {
							g.getCentroid().getX(),
							g.getCentroid().getY(),
							totPop,
							(d[1]+d[2]+d[3]+d[4]+d[5])/totPop, // 0 to 24
							(d[14]+d[15]+d[16]+d[17]+d[18])/totPop, // 65 older
							//d[19], // median age
							d[76]/totPop, // white
							d[77]/totPop, // black
							d[79]/totPop, // asian
							d[110]/totPop, // hispanic
							//d[111]/d[110], // white hispanic
							//d[126]/totPop, // in households
							//d[145]/d[144], // family households
							d[161], // avg household size
							//d[162], // avg family size
							d[176]/d[172], // renter-occupied housing units	
					};
					
					if( totPop == 0 )
						continue;
					
					nGeoms.add(g);
										
					nSamples.add(nd);
				}
			}		
			String[] names = new String[]{
					"x",
					"y",
					"pop",
					"0to24",
					"65older",
					//"mAge",
					"white",
					"black",
					"asian",
					"hispanic",
					//"whiteHis",
					//"inHH",
					//"famHH",
					"avgHHs",
					//"avgFams",
					"rentOcc",
			};
						
			DataUtils.writeShape(nSamples, nGeoms, names, targetCRS, "output/chicago.shp");
			DataUtils.writeCSV("output/chicago.csv", nSamples, names);
		} catch (MismatchedDimensionException e) {
			e.printStackTrace();
		} catch (NoSuchAuthorityCodeException e) {
			e.printStackTrace();
		} catch (FactoryException e) {
			e.printStackTrace();
		} finally {
		}
	}
}

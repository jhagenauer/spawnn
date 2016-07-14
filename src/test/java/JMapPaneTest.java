import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.map.FeatureLayer;
import org.geotools.map.Layer;
import org.geotools.map.MapContent;
import org.geotools.map.MapViewport;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.geotools.styling.Symbolizer;
import org.geotools.swing.JMapPane;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;

public class JMapPaneTest extends JFrame {
	
	public JMapPane mp;
	
	public JMapPaneTest() {
		mp = new JMapPane();		
		mp.setMapContent(new MapContent());
		add(mp);
		setSize(400,400);
		setVisible(true);
	}
	
	public void updateMP() {
		
		MapContent mc = mp.getMapContent();		
						
		for( int j = 0; j < 10; j++ ) {
			GeometryFactory gf = new GeometryFactory();
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("Points");
			typeBuilder.add("the_geom",Point.class);
			
			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
			Random r = new Random();		
			DefaultFeatureCollection fc = new DefaultFeatureCollection();
			for( int i = 0; i < 100000; i++ ) {
				Point p = gf.createPoint(new Coordinate(r.nextDouble(),r.nextDouble()));
				featureBuilder.set("the_geom", p);
				fc.add( featureBuilder.buildFeature(""+fc.size()));
			}

			Symbolizer sym = new StyleBuilder().createPointSymbolizer();
			mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sym)));
		}
	}
	
	public static void main(String[] args) {
		JMapPaneTest t = new JMapPaneTest();
		for( int i = 0; i < 20; i++ ) {
			System.out.println(i);
			t.updateMP();
		}
		System.exit(1);
	}
}

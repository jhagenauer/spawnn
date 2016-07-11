import java.util.Random;

import javax.swing.JFrame;

import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.map.MapViewport;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
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
		
		updateMP();
		
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		setSize(800, 800);
		
		mp.setSize(getSize());
	}
	
	public void updateMP() {
		GeometryFactory gf = new GeometryFactory();
		SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
		typeBuilder.setName("Points");
		typeBuilder.add("the_geom",Point.class);
		
		SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
		Random r = new Random();		
		DefaultFeatureCollection fc = new DefaultFeatureCollection();
		for( int i = 0; i < 1000; i++ ) {
			Point p = gf.createPoint(new Coordinate(r.nextDouble(),r.nextDouble()));
			featureBuilder.set("the_geom", p);
			fc.add( featureBuilder.buildFeature(""+fc.size()));
		}

		MapContent mc = mp.getMapContent();
		Symbolizer sym = new StyleBuilder().createPointSymbolizer();
		mc.layers().clear();
		mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sym)));
		mc.setViewport( new MapViewport(fc.getBounds()));
	}
	
	public static void main(String[] args) {
		JMapPaneTest t = new JMapPaneTest();
	}
}

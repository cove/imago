AssumeFPS(30000,1001)
AssumeBFF()
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=0.9828,SourceMatch=3,Lossless=2,TR2=3,MatchEdi="NNEDI3",MatchEdi2="NNEDI3",FPSDivisor=2)

# Remove white horizontal streaks near top of frame (rotate so DeScratch sees them as vertical)
TurnLeft()
DeScratch(mindif=3, maxgap=10, maxwidth=5, modeY=2, minlen=100, maxangle=0, left=0, right=24)
TurnRight()

# stabilization disabled
Crop(4,2,-8,-10)
LanczosResize(640,480)
ConvertToYV12(interlaced=false)
# after deinterlace / capture / conversion
SmoothLevels(16, 1.0, 255, 16, 235, limiter=1, tvrange=true, dither=0)

# optional wobble fix for text edges (set to true on problem tapes)
function WobbleFix(clip c) { c.TurnLeft().Blur(0.3).TurnRight() }
wobble_fix = false
c = last
if (wobble_fix) {
    c = WobbleFix(c)
}
return c

import streamlit as st

st.title("WAPA Image Slider HTML Generator")
st.write("Generate YM-compatible carousel HTML from image URLs and links.")

# --- Number of slides ---
num_slides = st.number_input("How many slides?", min_value=1, max_value=5, value=3)

slides = []

for i in range(num_slides):
    st.subheader(f"Slide {i+1}")
    img = st.text_input(f"Image URL for Slide {i+1}", key=f"img_{i}")
    link = st.text_input(f"Optional click-through URL for Slide {i+1}", key=f"link_{i}")
    alt = st.text_input(f"Alt text for Slide {i+1}", key=f"alt_{i}", value=f"Slide {i+1}")
    slides.append({"img": img, "link": link, "alt": alt})

# --- Generate HTML ---
if st.button("Generate HTML"):
    html = []

    html.append('<div id="myCarousel" class="carousel slide carousel-fade" '
                'data-ride="carousel" data-interval="5000">')

    # Indicators
    html.append('  <ol class="carousel-indicators">')
    for i in range(num_slides):
        active = ' class="active"' if i == 0 else ""
        html.append(f'    <li data-target="#myCarousel" data-slide-to="{i}"{active}></li>')
    html.append('  </ol>')

    # Slides
    html.append('  <div class="carousel-inner" role="listbox">')
    for i, slide in enumerate(slides):
        active = " active" if i == 0 else ""
        html.append(f'    <div class="item{active}" style="text-align:center;">')

        if slide["link"]:
            html.append(f'      <a href="{slide["link"]}" target="_blank">')

        html.append(f'        <img alt="{slide["alt"]}" src="{slide["img"]}" />')

        if slide["link"]:
            html.append("      </a>")

        html.append("    </div>")

    html.append("  </div>")

    # Controls
    html.append('  <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">')
    html.append('    <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>')
    html.append('    <span class="sr-only">Previous</span>')
    html.append("  </a>")
    html.append('  <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">')
    html.append('    <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>')
    html.append('    <span class="sr-only">Next</span>')
    html.append("  </a>")

    html.append("</div>")

    final_html = "\n".join(html)

    st.success("HTML Generated!")
    st.code(final_html, language="html")


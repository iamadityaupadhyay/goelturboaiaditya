from django import forms

class InfluenceForm(forms.Form):
    posts = forms.CharField(label='Posts', max_length=100)
    followers = forms.CharField(label='Followers', max_length=100)
    avg_likes = forms.CharField(label='Avg Likes', max_length=100)
    sixty_day_eng_rate = forms.CharField(label='60 Day Engagement Rate', max_length=100)
    new_post_avg_like = forms.CharField(label='New Post Avg Like', max_length=100)
    total_likes = forms.CharField(label='Total Likes', max_length=100)

